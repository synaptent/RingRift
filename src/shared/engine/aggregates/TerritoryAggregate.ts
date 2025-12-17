/**
 * ═══════════════════════════════════════════════════════════════════════════
 * TerritoryAggregate - Consolidated Territory Domain
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This aggregate consolidates all territory disconnection detection, border
 * analysis, processability checks, and collapse/elimination logic from:
 *
 * - territoryDetection.ts → disconnection detection and region finding
 * - territoryBorders.ts → border marker computation
 * - territoryDecisionHelpers.ts → region processing decisions
 * - territoryProcessing.ts → territory processing orchestration
 * - validators/TerritoryValidator.ts → validation
 * - mutators/TerritoryMutator.ts → mutation
 *
 * Rule Reference: Section 12 - Territory Processing
 *
 * Key Rules:
 * - RR-CANON-R120: Territory disconnection (capture may split markers into regions)
 * - RR-CANON-R121: Mini-region definition (region smaller than minimum threshold)
 * - RR-CANON-R122: Mini-region processing (player chooses to remove or keep)
 * - RR-CANON-R123: Self-elimination (player may choose to self-eliminate)
 * - RR-CANON-R124: Border markers used to identify region boundaries
 * - Process all disconnected regions before checking victory
 *
 * **Self-Elimination Cost Rules (§12.2, RR-CANON-R022, RR-CANON-R145):**
 *
 * TERRITORY PROCESSING COST:
 * - Player must eliminate an **entire stack cap** (all consecutive top rings of
 *   their colour) from a controlled stack outside the processed region.
 * - **All controlled stacks are eligible, including height-1 standalone rings.**
 *   Per RR-CANON-R022 and RR-CANON-R145.
 *
 * FORCED ELIMINATION COST (RR-CANON-R070, RR-CANON-R100):
 * - Player must eliminate an **entire stack cap** from **any** controlled stack.
 * - **All controlled stacks are eligible, including height-1 standalone rings.**
 *
 * RECOVERY ACTION EXCEPTION:
 * - When territory processing is triggered by a recovery action, the cost is
 *   only 1 buried ring extraction (bottommost ring from a chosen stack),
 *   not an entire cap.
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Backward compatibility: Source files continue to export their functions
 */

import type {
  GameState,
  BoardState,
  Position,
  Move,
  Territory,
  AdjacencyType,
} from '../../types/game';
import { BOARD_CONFIGS, positionToString } from '../../types/game';

import type { ProcessTerritoryAction, EliminateStackAction } from '../types';

import { SQUARE_MOORE_DIRECTIONS } from '../core';
import {
  eliminateFromStack,
  isStackEligibleForElimination,
  getRingsToEliminate,
  calculateCapHeight,
  type EliminationContext,
} from './EliminationAggregate';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../territoryDetection';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Disconnected region detected on the board.
 * Represents a set of connected spaces that form a territory region.
 */
export interface DisconnectedRegion extends Territory {
  /** Positions that form this region */
  spaces: Position[];
  /** Player who controls this region (0 if unassigned) */
  controllingPlayer: number;
  /** Whether this is a disconnected region */
  isDisconnected: boolean;
}

/**
 * Decision about how to process a territory region.
 */
export interface TerritoryProcessingDecision {
  /** Type of territory decision */
  type: 'process_region' | 'eliminate_from_stack';
  /** Player making the decision */
  player: number;
  /** The region being processed (for process_region) */
  region?: DisconnectedRegion;
  /** Stack position for elimination (for eliminate_from_stack) */
  stackPosition?: Position;
  /** Number of rings to eliminate */
  eliminationCount?: number;
}

/**
 * Result of validating a territory-related action.
 */
export type TerritoryValidationResult = {
  valid: boolean;
  reason?: string;
  code?: string;
};

/**
 * Result of applying a territory mutation.
 */
export type TerritoryMutationResult =
  | { success: true; newState: GameState }
  | { success: false; reason: string };

/**
 * Minimal context required to process a region on a given board.
 */
export interface TerritoryProcessingContext {
  /** Player who is claiming the region and receiving credited eliminations. */
  player: number;
  /** Optional board type override */
  boardType?: string;
  /**
   * Self-elimination context for territory processing.
   *
   * - 'territory' (default): cap elimination from an eligible P-controlled stack outside the region.
   * - 'recovery': buried-ring extraction from any stack outside the region that contains a buried ring of P.
   */
  eliminationContext?: 'territory' | 'recovery';
}

/**
 * Structured result of applying territory processing for a single region.
 */
export interface TerritoryProcessingOutcome {
  /**
   * Updated board after applying internal eliminations and collapsing region
   * spaces + border markers.
   */
  board: BoardState;

  /** The region that was processed. */
  region: Territory;

  /** Border marker positions that were collapsed. */
  borderMarkers: Position[];

  /** Per-player eliminated-ring deltas caused by internal eliminations. */
  eliminatedRingsByPlayer: { [player: number]: number };

  /** Per-player territory-space gains caused by collapsing. */
  territoryGainedByPlayer: { [player: number]: number };
}

/**
 * Options that control how territory-processing moves are enumerated.
 */
export interface TerritoryEnumerationOptions {
  /**
   * Whether to derive regions from `state.board.territories` or re-run the
   * shared detector.
   */
  detectionMode?: 'use_board_cache' | 'detect_now';

  /**
   * Optional test-only override list of disconnected regions.
   */
  testOverrideRegions?: Territory[];

  /** Whether to filter by processability */
  filterByProcessability?: boolean;
}

/**
 * Result of applying a `choose_territory_option` decision (legacy alias: `process_territory_region`).
 */
export interface TerritoryProcessApplicationOutcome {
  /** Next GameState after applying the region-processing consequences. */
  nextState: GameState;

  /** Identifier of the processed region. */
  processedRegionId: string;

  /** The processed region. */
  processedRegion: Territory;

  /** True when mandatory self-elimination is required. */
  pendingSelfElimination: boolean;
}

/**
 * Scope for elimination decisions.
 */
export interface TerritoryEliminationScope {
  /** Optional processed region identifier */
  processedRegionId?: string;
  /**
   * Optional processed region geometry. When present, elimination targets
   * are restricted to stacks outside this region.
   */
  processedRegion?: Territory;
  /**
   * Elimination context for territory self-elimination.
   * - 'territory': cap elimination (RR-CANON-R145)
   * - 'recovery': buried-ring extraction (RR-CANON-R114)
   */
  eliminationContext?: 'territory' | 'recovery';
  /** Optional stack position to eliminate from */
  stackPosition?: Position;
  /** Maximum rings to eliminate */
  maxRings?: number;
}

/**
 * Result of applying an `eliminate_rings_from_stack` decision.
 */
export interface EliminateRingsFromStackOutcome {
  /** Next GameState after eliminating rings from the chosen stack. */
  nextState: GameState;
}

/**
 * Mode for border marker computation.
 */
export type TerritoryBorderMode =
  | 'ts_legacy'
  | 'rust_aligned'
  | 'own_markers'
  | 'opponent_markers'
  | 'all_markers';

/**
 * Options for border marker computation.
 */
export interface TerritoryBorderOptions {
  mode?: TerritoryBorderMode;
  player?: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers - Position & Board Utilities
// ═══════════════════════════════════════════════════════════════════════════

function isValidPosition(position: Position, board: BoardState): boolean {
  const size = board.size;
  if (board.type === 'hexagonal' || board.type === 'hex8') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z || -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  } else {
    return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
  }
}

function cloneBoard(board: BoardState): BoardState {
  return {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };
}

function computeNextMoveNumber(state: GameState): number {
  if (state.history && state.history.length > 0) {
    const last = state.history[state.history.length - 1];
    if (typeof last.moveNumber === 'number' && last.moveNumber > 0) {
      return last.moveNumber + 1;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastLegacy = state.moveHistory[state.moveHistory.length - 1];
    if (typeof lastLegacy.moveNumber === 'number' && lastLegacy.moveNumber > 0) {
      return lastLegacy.moveNumber + 1;
    }
  }

  return 1;
}

function didCurrentTurnIncludeRecoverySlide(state: GameState, player: number): boolean {
  const history = state.moveHistory;
  for (let i = history.length - 1; i >= 0; i--) {
    const move = history[i];
    if (move.player !== player) {
      break;
    }
    if (move.type === 'recovery_slide') {
      return true;
    }
  }
  return false;
}

function getPendingTerritorySelfEliminationRegion(
  state: GameState,
  player: number
): Territory | null {
  const lastMove =
    state.moveHistory.length > 0 ? state.moveHistory[state.moveHistory.length - 1] : undefined;
  if (!lastMove || lastMove.player !== player) {
    return null;
  }

  const isTerritoryRegionMove =
    lastMove.type === 'choose_territory_option' || lastMove.type === 'process_territory_region';
  if (!isTerritoryRegionMove) {
    return null;
  }

  const regions = lastMove.disconnectedRegions;
  if (!regions || regions.length === 0) {
    return null;
  }
  return regions[0] ?? null;
}

function deriveTerritoryEliminationContext(
  state: GameState,
  player: number
): 'territory' | 'recovery' {
  return didCurrentTurnIncludeRecoverySlide(state, player) ? 'recovery' : 'territory';
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers - Border Markers
// ═══════════════════════════════════════════════════════════════════════════

function getTerritoryNeighbors(
  board: BoardState,
  position: Position,
  adjacencyType: AdjacencyType
): Position[] {
  const neighbors: Position[] = [];
  const { x, y, z } = position;

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
      const neighbor: Position = {
        x: x + dir.x,
        y: y + dir.y,
        z: (z ?? 0) + dir.z,
      };
      if (isValidPosition(neighbor, board)) {
        neighbors.push(neighbor);
      }
    }
    return neighbors;
  }

  if (adjacencyType === 'von_neumann') {
    const directions = [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: -1, y: 0 },
    ];
    for (const dir of directions) {
      const neighbor: Position = { x: x + dir.x, y: y + dir.y };
      if (isValidPosition(neighbor, board)) {
        neighbors.push(neighbor);
      }
    }
    return neighbors;
  }

  // Fallback: Moore adjacency on square boards
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      if (dx === 0 && dy === 0) continue;
      const neighbor: Position = { x: x + dx, y: y + dy };
      if (isValidPosition(neighbor, board)) {
        neighbors.push(neighbor);
      }
    }
  }

  return neighbors;
}

function getMooreNeighbors(board: BoardState, position: Position): Position[] {
  const neighbors: Position[] = [];

  // Moore adjacency is only meaningful on square boards
  if (board.type === 'hexagonal' || board.type === 'hex8') {
    return neighbors;
  }

  for (const dir of SQUARE_MOORE_DIRECTIONS) {
    const neighbor: Position = {
      x: position.x + dir.x,
      y: position.y + dir.y,
    };
    if (isValidPosition(neighbor, board)) {
      neighbors.push(neighbor);
    }
  }

  return neighbors;
}

function comparePositionsStable(a: Position, b: Position): number {
  // For pure square boards, sort row-major (y, then x).
  if (a.z === undefined && b.z === undefined) {
    return a.y - b.y || a.x - b.x;
  }

  // Cube-lexicographic ordering for hex coordinates (x, then y, then z).
  const az = a.z ?? -a.x - a.y;
  const bz = b.z ?? -b.x - b.y;
  return a.x - b.x || a.y - b.y || az - bz;
}

// ═══════════════════════════════════════════════════════════════════════════
// Detection Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Find all disconnected regions on the board.
 * Rule Reference: Section 12.2 - Territory Disconnection
 *
 * Delegates to the shared territoryDetection module for core region detection
 * logic, then applies optional player filtering.
 *
 * @param state GameState to analyze
 * @param player Optional: filter for regions relevant to this player
 * @returns Array of disconnected regions
 */
export function findDisconnectedRegions(state: GameState, player?: number): DisconnectedRegion[] {
  const board = state.board;

  // Delegate to shared territory detection for core region finding
  const baseRegions = findDisconnectedRegionsShared(board);

  // Cast Territory[] to DisconnectedRegion[] (compatible structure)
  const disconnectedRegions: DisconnectedRegion[] = baseRegions.map((region) => ({
    ...region,
  }));

  // Filter by player if specified
  if (player !== undefined) {
    return disconnectedRegions.filter((region) => {
      // Region is relevant if player has stacks in it
      return region.spaces.some((pos) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        return stack && stack.controllingPlayer === player;
      });
    });
  }

  return disconnectedRegions;
}

/**
 * Compute border marker positions for a disconnected region.
 *
 * @param state GameState containing board
 * @param region The disconnected region
 * @returns Array of border marker positions
 */
export function computeBorderMarkers(state: GameState, region: DisconnectedRegion): Position[] {
  return getBorderMarkerPositionsForRegion(state.board, region.spaces, {
    mode: 'rust_aligned',
  });
}

/**
 * Get border marker positions for a territory region.
 *
 * @param board The board state
 * @param regionSpaces Positions forming the region
 * @param opts Border computation options
 * @returns Array of border marker positions in stable order
 */
export function getBorderMarkerPositionsForRegion(
  board: BoardState,
  regionSpaces: Position[],
  opts?: TerritoryBorderOptions
): Position[] {
  const mode = opts?.mode ?? 'rust_aligned';

  if (regionSpaces.length === 0) {
    return [];
  }

  const regionSet = new Set(regionSpaces.map((p) => positionToString(p)));
  const config = BOARD_CONFIGS[board.type];
  const territoryAdjacency: AdjacencyType = config.territoryAdjacency;

  // Step 1: seed border markers = direct territory-adjacent markers.
  const seedMap = new Map<string, Position>();

  for (const space of regionSpaces) {
    const neighbors = getTerritoryNeighbors(board, space, territoryAdjacency);
    for (const neighbor of neighbors) {
      const key = positionToString(neighbor);
      if (regionSet.has(key)) continue;
      const marker = board.markers.get(key);
      if (marker && !seedMap.has(key)) {
        seedMap.set(key, neighbor);
      }
    }
  }

  if (seedMap.size === 0) {
    return [];
  }

  // Step 2: expand across connected markers using Moore adjacency on square boards
  const borderMarkers = new Map<string, Position>(seedMap);

  if (board.type !== 'hexagonal' && (mode === 'rust_aligned' || mode === 'ts_legacy')) {
    const queue: Position[] = Array.from(seedMap.values());
    const visited = new Set<string>(seedMap.keys());

    while (queue.length > 0) {
      const current = queue.shift();
      if (!current) continue;
      const neighbors = getMooreNeighbors(board, current);

      for (const neighbor of neighbors) {
        const key = positionToString(neighbor);
        if (visited.has(key)) continue;
        if (regionSet.has(key)) continue; // never step into region

        const marker = board.markers.get(key);
        if (marker) {
          visited.add(key);
          borderMarkers.set(key, neighbor);
          queue.push(neighbor);
        }
      }
    }
  }

  const result = Array.from(borderMarkers.values());
  result.sort(comparePositionsStable);
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Validate a territory processing decision.
 *
 * @param state Current game state
 * @param decision The decision to validate
 * @returns Validation result
 */
export function validateTerritoryDecision(
  state: GameState,
  decision: TerritoryProcessingDecision
): TerritoryValidationResult {
  if (decision.type === 'process_region') {
    const action: ProcessTerritoryAction = {
      type: 'PROCESS_TERRITORY',
      playerId: decision.player,
      regionId: decision.region ? positionToString(decision.region.spaces[0]) : '',
    };
    return validateProcessTerritory(state, action);
  } else if (decision.type === 'eliminate_from_stack') {
    if (!decision.stackPosition) {
      return { valid: false, reason: 'Stack position required', code: 'MISSING_POSITION' };
    }
    const action: EliminateStackAction = {
      type: 'ELIMINATE_STACK',
      playerId: decision.player,
      stackPosition: decision.stackPosition,
    };
    return validateEliminateStack(state, action);
  }

  return { valid: false, reason: 'Unknown decision type', code: 'UNKNOWN_TYPE' };
}

/**
 * Validate a PROCESS_TERRITORY action.
 */
export function validateProcessTerritory(
  state: GameState,
  action: ProcessTerritoryAction
): TerritoryValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'territory_processing') {
    return { valid: false, reason: 'Not in territory processing phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Region Existence Check
  const region = state.board.territories.get(action.regionId);
  if (!region) {
    return { valid: false, reason: 'Region not found', code: 'REGION_NOT_FOUND' };
  }

  // 4. Disconnection Check
  if (!region.isDisconnected) {
    return { valid: false, reason: 'Region is not disconnected', code: 'REGION_NOT_DISCONNECTED' };
  }

  return { valid: true };
}

/**
 * Validate an ELIMINATE_STACK action.
 */
export function validateEliminateStack(
  state: GameState,
  action: EliminateStackAction
): TerritoryValidationResult {
  // 1. Phase check
  if (state.currentPhase !== 'territory_processing') {
    return { valid: false, reason: 'Not in territory processing phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Stack existence and ownership
  const key = positionToString(action.stackPosition);
  const stack = state.board.stacks.get(key);

  if (!stack) {
    return { valid: false, reason: 'Stack not found', code: 'STACK_NOT_FOUND' };
  }

  if (stack.controllingPlayer !== action.playerId) {
    return { valid: false, reason: 'Stack is not controlled by player', code: 'NOT_YOUR_STACK' };
  }

  if (stack.stackHeight <= 0) {
    return { valid: false, reason: 'Stack is empty', code: 'EMPTY_STACK' };
  }

  return { valid: true };
}

// ═══════════════════════════════════════════════════════════════════════════
// Enumeration Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enumerate territory processing decisions available to a player.
 *
 * @param state Current game state
 * @param region The disconnected region to process
 * @returns Array of possible decisions
 */
export function enumerateTerritoryDecisions(
  state: GameState,
  region: DisconnectedRegion
): TerritoryProcessingDecision[] {
  const decisions: TerritoryProcessingDecision[] = [];

  // The main decision is whether to process the region
  decisions.push({
    type: 'process_region',
    player: state.currentPlayer,
    region,
  });

  return decisions;
}

/**
 * Enumerate `choose_territory_option` decision moves for the specified player.
 */
export function enumerateProcessTerritoryRegionMoves(
  state: GameState,
  player: number,
  options?: TerritoryEnumerationOptions
): Move[] {
  const board = state.board;
  const territoryEliminationContext = deriveTerritoryEliminationContext(state, player);

  // Canonical ordering: when a territory region has been processed, the next
  // required action is mandatory self-elimination via eliminate_rings_from_stack
  // before processing any additional regions (RR-CANON-R145).
  if (state.currentPhase === 'territory_processing') {
    const pendingRegion = getPendingTerritorySelfEliminationRegion(state, player);
    if (pendingRegion) {
      return [];
    }
  }

  const ctx: TerritoryProcessingContext = {
    player,
    eliminationContext: territoryEliminationContext,
  };
  const overrideRegions = options?.testOverrideRegions;

  const processableRegions =
    overrideRegions && overrideRegions.length > 0
      ? filterProcessableTerritoryRegions(board, overrideRegions, ctx)
      : getProcessableTerritoryRegions(board, ctx);

  if (processableRegions.length === 0) {
    return [];
  }

  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  processableRegions.forEach((region, index) => {
    if (!region.spaces || region.spaces.length === 0) {
      return;
    }

    const representative = region.spaces[0];
    const regionKey = representative ? positionToString(representative) : `region-${index}`;

    moves.push({
      id: `process-region-${index}-${regionKey}`,
      type: 'choose_territory_option',
      player,
      to: representative ?? { x: 0, y: 0 },
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  });

  return moves;
}

/**
 * Enumerate `eliminate_rings_from_stack` decision moves for territory processing.
 *
 * Canonical ordering (RR-CANON-R145 / RR-CANON-R114):
 * - Player selects a region to process via `choose_territory_option`.
 * - Player must then immediately pay the mandatory self-elimination cost via
 *   `eliminate_rings_from_stack` before processing any additional regions.
 *
 * This helper returns elimination moves only when that self-elimination is
 * currently pending (i.e., the most recent move by `player` was a territory
 * region decision in `territory_processing`).
 */
export function enumerateTerritoryEliminationMoves(
  state: GameState,
  player: number,
  scope?: TerritoryEliminationScope
): Move[] {
  if (state.currentPhase !== 'territory_processing') {
    return [];
  }

  const eliminationContext: EliminationContext =
    scope?.eliminationContext ?? deriveTerritoryEliminationContext(state, player);

  const processedRegion =
    scope?.processedRegion ?? getPendingTerritorySelfEliminationRegion(state, player);

  if (!processedRegion) {
    return [];
  }

  const processedRegionKeySet = new Set(processedRegion.spaces.map((p) => positionToString(p)));
  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  for (const [key, stack] of state.board.stacks.entries()) {
    if (processedRegionKeySet.has(key)) {
      continue;
    }

    if (eliminationContext !== 'recovery' && stack.controllingPlayer !== player) {
      continue;
    }

    const eligibility = isStackEligibleForElimination(stack, eliminationContext, player);
    if (!eligibility.eligible) {
      continue;
    }

    const capHeight = calculateCapHeight(stack.rings);
    if (capHeight <= 0) {
      continue;
    }

    const ringsToEliminate = getRingsToEliminate(stack, eliminationContext);

    moves.push({
      id: `eliminate-${key}`,
      type: 'eliminate_rings_from_stack',
      player,
      to: stack.position,
      eliminatedRings: [{ player, count: ringsToEliminate }],
      eliminationFromStack: {
        position: stack.position,
        capHeight,
        totalHeight: stack.stackHeight,
      },
      eliminationContext,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  }

  return moves;
}

// ═══════════════════════════════════════════════════════════════════════════
// Processability Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a territory region can be processed by a player.
 *
 * Self-elimination prerequisite (FAQ Q23 / §12.2):
 * A disconnected region is processable for ctx.player iff that player
 * controls at least one ELIGIBLE stack/cap outside the region.
 *
 * Eligibility per RR-CANON-R022/R145: ALL controlled stacks are eligible,
 * including height-1 standalone rings.
 *
 * DELEGATES TO EliminationAggregate for canonical eligibility check.
 */
export function canProcessTerritoryRegion(
  board: BoardState,
  region: Territory,
  ctx: TerritoryProcessingContext
): boolean {
  const regionKeySet = new Set(region.spaces.map((p) => positionToString(p)));
  const eliminationContext = ctx.eliminationContext ?? 'territory';

  for (const [key, stack] of board.stacks.entries()) {
    if (!regionKeySet.has(key)) {
      if (eliminationContext === 'recovery') {
        // Recovery-context territory processing (RR-CANON-R114): the stack need
        // not be controlled by ctx.player; it must contain a buried ring of
        // ctx.player that can be extracted.
        const eligibility = isStackEligibleForElimination(stack, 'recovery', ctx.player);
        if (eligibility.eligible) {
          return true;
        }
        continue;
      }

      // Normal territory context (RR-CANON-R145): must be an eligible cap
      // target controlled by ctx.player.
      if (stack.controllingPlayer !== ctx.player) {
        continue;
      }
      const eligibility = isStackEligibleForElimination(stack, 'territory', ctx.player);
      if (eligibility.eligible) {
        return true;
      }
      // Stack not eligible (eligibility check failed)
    }
  }

  // No eligible cap targets for this player outside the region.
  return false;
}

/**
 * Filter regions to those processable by the given player.
 */
export function filterProcessableTerritoryRegions(
  board: BoardState,
  regions: Territory[],
  ctx: TerritoryProcessingContext
): Territory[] {
  if (regions.length === 0) {
    return [];
  }

  return regions.filter((region) => canProcessTerritoryRegion(board, region, ctx));
}

/**
 * Get all processable territory regions for a player.
 *
 * Delegates to the shared territoryDetection module for core region detection.
 */
export function getProcessableTerritoryRegions(
  board: BoardState,
  ctx: TerritoryProcessingContext
): Territory[] {
  // Delegate to shared territory detection
  const allRegions = findDisconnectedRegionsShared(board);

  return filterProcessableTerritoryRegions(board, allRegions, ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply a territory processing decision and return a mutation result.
 */
export function applyTerritoryDecision(
  state: GameState,
  decision: TerritoryProcessingDecision
): TerritoryMutationResult {
  try {
    const validation = validateTerritoryDecision(state, decision);
    if (!validation.valid) {
      return {
        success: false,
        reason: validation.reason ?? 'Invalid territory decision',
      };
    }

    if (decision.type === 'process_region' && decision.region) {
      const move: Move = {
        id: `process-region-${positionToString(decision.region.spaces[0])}`,
        type: 'choose_territory_option',
        player: decision.player,
        to: decision.region.spaces[0],
        disconnectedRegions: [decision.region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: computeNextMoveNumber(state),
      };

      const outcome = applyProcessTerritoryRegionDecision(state, move);
      return { success: true, newState: outcome.nextState };
    } else if (decision.type === 'eliminate_from_stack' && decision.stackPosition) {
      const move: Move = {
        id: `eliminate-${positionToString(decision.stackPosition)}`,
        type: 'eliminate_rings_from_stack',
        player: decision.player,
        to: decision.stackPosition,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: computeNextMoveNumber(state),
      };

      const outcome = applyEliminateRingsFromStackDecision(state, move);
      return { success: true, newState: outcome.nextState };
    }

    return { success: false, reason: 'Invalid decision data' };
  } catch (error) {
    return {
      success: false,
      reason: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Apply territory region collapse at board level.
 */
export function applyTerritoryRegion(
  board: BoardState,
  region: Territory,
  ctx: TerritoryProcessingContext
): TerritoryProcessingOutcome {
  const nextBoard = cloneBoard(board);

  // 1. Determine border markers for this region
  const borderMarkers = getBorderMarkerPositionsForRegion(board, region.spaces, {
    mode: 'rust_aligned',
  });

  const regionKeySet = new Set(region.spaces.map((p) => positionToString(p)));

  // 2. Eliminate all stacks inside the region before collapsing spaces.
  let internalEliminations = 0;

  for (const pos of region.spaces) {
    const key = positionToString(pos);
    const stack = nextBoard.stacks.get(key);
    if (!stack) {
      continue;
    }

    internalEliminations += stack.stackHeight;
    nextBoard.stacks.delete(key);
  }

  // 3. Collapse all spaces in the region to ctx.player's colour.
  for (const pos of region.spaces) {
    const key = positionToString(pos);
    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, ctx.player);
  }

  // 4. Collapse all border markers to ctx.player's colour.
  for (const pos of borderMarkers) {
    const key = positionToString(pos);
    if (regionKeySet.has(key)) {
      continue; // Skip interior spaces
    }

    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, ctx.player);
  }

  // Territory gain is |region.spaces| + |borderMarkers|
  const territoryGain = region.spaces.length + borderMarkers.length;

  const territoryGainedByPlayer: { [player: number]: number } = {};
  if (territoryGain > 0) {
    territoryGainedByPlayer[ctx.player] = territoryGain;
  }

  // 5. Credit all internal eliminations to ctx.player
  const eliminatedRingsByPlayer: { [player: number]: number } = {};
  if (internalEliminations > 0) {
    eliminatedRingsByPlayer[ctx.player] = internalEliminations;

    const updatedElims = { ...nextBoard.eliminatedRings };
    updatedElims[ctx.player] = (updatedElims[ctx.player] || 0) + internalEliminations;
    nextBoard.eliminatedRings = updatedElims;
  }

  return {
    board: nextBoard,
    region,
    borderMarkers,
    eliminatedRingsByPlayer,
    territoryGainedByPlayer,
  };
}

/**
 * Apply a `choose_territory_option` move (legacy alias: `process_territory_region`).
 */
export function applyProcessTerritoryRegionDecision(
  state: GameState,
  move: Move
): TerritoryProcessApplicationOutcome {
  if (move.type !== 'choose_territory_option' && move.type !== 'process_territory_region') {
    throw new Error(
      `applyProcessTerritoryRegionDecision expected move.type === 'choose_territory_option' (or legacy 'process_territory_region'), got '${move.type}'`
    );
  }

  const player = move.player;
  const territoryEliminationContext = deriveTerritoryEliminationContext(state, player);

  // Prefer the concrete Territory attached to the Move
  let region: Territory | undefined =
    move.disconnectedRegions && move.disconnectedRegions.length > 0
      ? move.disconnectedRegions[0]
      : undefined;

  // Fallback: re-derive a processable region
  if (!region) {
    const candidates = getProcessableTerritoryRegions(state.board, {
      player,
      eliminationContext: territoryEliminationContext,
    });
    if (candidates.length === 1) {
      region = candidates[0];
    } else if (candidates.length > 1) {
      if (move.to) {
        const repKey = positionToString(move.to);
        region = candidates.find(
          (r) => r.spaces.length > 0 && positionToString(r.spaces[0]) === repKey
        );
      }

      if (!region && move.id && move.id.startsWith('process-region-')) {
        const tail = move.id.slice('process-region-'.length);
        const dashIndex = tail.indexOf('-');
        const regionKeyStr = dashIndex >= 0 ? tail.slice(dashIndex + 1) : tail;
        region = candidates.find(
          (r) => r.spaces.length > 0 && positionToString(r.spaces[0]) === regionKeyStr
        );
      }
    }
  }

  if (!region) {
    return {
      nextState: state,
      processedRegionId: move.id,
      processedRegion: {
        spaces: [],
        controllingPlayer: player,
        isDisconnected: true,
      },
      pendingSelfElimination: false,
    };
  }

  // Canonical guard: controllingPlayer should never be 0 (neutral) for a
  // processable disconnected region. Such regions indicate a non-canonical
  // recording or territory detection bug and must fail fast.
  if (region.controllingPlayer === 0) {
    throw new Error(
      `Non-canonical territory region: controllingPlayer=0 for choose_territory_option (move ${move.id ?? ''})`
    );
  }

  // Enforce the self-elimination prerequisite
  if (
    !canProcessTerritoryRegion(state.board, region, {
      player,
      eliminationContext: territoryEliminationContext,
    })
  ) {
    const representative = region.spaces[0];
    const regionKey = representative ? positionToString(representative) : 'region-0';
    const processedRegionId =
      move.id && move.id.length > 0 ? move.id : `process-region-0-${regionKey}`;

    return {
      nextState: state,
      processedRegionId,
      processedRegion: region,
      pendingSelfElimination: false,
    };
  }

  const ctx: TerritoryProcessingContext = { player };
  const outcome = applyTerritoryRegion(state.board, region, ctx);

  // Project board-level deltas into GameState-level aggregates
  const territoryGain = outcome.territoryGainedByPlayer[player] ?? 0;
  const internalElims = outcome.eliminatedRingsByPlayer[player] ?? 0;

  let nextPlayers = state.players.map((p) => ({ ...p }));

  if (territoryGain > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === player ? { ...p, territorySpaces: p.territorySpaces + territoryGain } : p
    );
  }

  if (internalElims > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === player ? { ...p, eliminatedRings: p.eliminatedRings + internalElims } : p
    );
  }

  const nextState: GameState = {
    ...state,
    board: outcome.board,
    players: nextPlayers,
    totalRingsEliminated: state.totalRingsEliminated + internalElims,
  };

  const representative = region.spaces[0];
  const regionKey = representative ? positionToString(representative) : 'region-0';
  const processedRegionId =
    move.id && move.id.length > 0 ? move.id : `process-region-0-${regionKey}`;

  return {
    nextState,
    processedRegionId,
    processedRegion: region,
    pendingSelfElimination: true,
  };
}

/**
 * Apply an `eliminate_rings_from_stack` or `forced_elimination` move.
 *
 * Both move types use the same ring elimination logic - the only difference
 * is semantic: `eliminate_rings_from_stack` is used during territory processing
 * as a player choice, while `forced_elimination` is used when a player has no
 * valid moves and must eliminate rings from a stack (RR-CANON-R070).
 *
 * DELEGATES TO EliminationAggregate for canonical elimination semantics.
 */
export function applyEliminateRingsFromStackDecision(
  state: GameState,
  move: Move
): EliminateRingsFromStackOutcome {
  if (move.type !== 'eliminate_rings_from_stack' && move.type !== 'forced_elimination') {
    throw new Error(
      `applyEliminateRingsFromStackDecision expected move.type === 'eliminate_rings_from_stack' or 'forced_elimination', got '${move.type}'`
    );
  }

  if (!move.to) {
    return { nextState: state };
  }

  const player = move.player;

  // Determine elimination context from move type and move data
  // - 'forced_elimination' move type: use 'forced' context (any stack eligible)
  // - 'eliminate_rings_from_stack': use move.eliminationContext or default to 'territory'
  const eliminationContext: EliminationContext =
    move.type === 'forced_elimination'
      ? 'forced'
      : ((move.eliminationContext as EliminationContext) ?? 'territory');

  // Delegate to canonical EliminationAggregate
  const eliminationResult = eliminateFromStack({
    context: eliminationContext,
    player,
    stackPosition: move.to,
    board: state.board,
  });

  if (!eliminationResult.success) {
    // Invalid or stale target; leave state unchanged defensively.
    return { nextState: state };
  }

  // Update players array (EliminationAggregate only updates board)
  const nextPlayers = state.players.map((p) =>
    p.playerNumber === player
      ? {
          ...p,
          eliminatedRings: p.eliminatedRings + eliminationResult.ringsEliminated,
        }
      : p
  );

  const nextState: GameState = {
    ...state,
    board: eliminationResult.updatedBoard,
    players: nextPlayers,
    totalRingsEliminated: state.totalRingsEliminated + eliminationResult.ringsEliminated,
  };

  return { nextState };
}

/**
 * Apply territory processing mutation via action.
 */
export function mutateProcessTerritory(
  state: GameState,
  action: ProcessTerritoryAction
): GameState {
  const newState = {
    ...state,
    board: cloneBoard(state.board),
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  const keptRegion = newState.board.territories.get(action.regionId);
  if (!keptRegion) throw new Error('TerritoryMutator: Kept region not found');

  // Mark kept region as connected
  keptRegion.isDisconnected = false;

  // Remove other disconnected regions
  for (const [id, region] of newState.board.territories) {
    if (
      region.controllingPlayer === action.playerId &&
      region.isDisconnected &&
      id !== action.regionId
    ) {
      for (const pos of region.spaces) {
        const key = positionToString(pos);
        newState.board.collapsedSpaces.delete(key);
      }

      newState.board.territories.delete(id);
    }
  }

  newState.lastMoveAt = new Date();
  return newState;
}

/**
 * Apply stack elimination mutation via action.
 * Per RR-CANON-R022, R122, R145, R100:
 * - 'line': Eliminate exactly ONE ring from the top (any controlled stack is eligible)
 * - 'territory': Eliminate entire cap (any controlled stack eligible, including height-1)
 * - 'forced': Eliminate entire cap (any controlled stack is eligible)
 */
export function mutateEliminateStack(state: GameState, action: EliminateStackAction): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
  };

  const key = positionToString(action.stackPosition);
  const stack = newState.board.stacks.get(key);

  if (!stack) {
    throw new Error('TerritoryAggregate: Stack to eliminate not found');
  }

  const capHeight = calculateCapHeight(stack.rings);
  const topRingOwner = stack.rings[0];

  // Determine how many rings to eliminate based on context (RR-CANON-R022, R122):
  // - 'line': Eliminate exactly ONE ring (per RR-CANON-R122)
  // - 'territory' or 'forced' or undefined: Eliminate entire cap (per RR-CANON-R145, R100)
  const eliminationContext = action.eliminationContext;
  const ringsToEliminate = eliminationContext === 'line' ? 1 : capHeight;

  // Remove rings from top
  const remainingRings = stack.rings.slice(ringsToEliminate);

  if (remainingRings.length === 0) {
    newState.board.stacks.delete(key);
  } else {
    stack.rings = remainingRings;
    stack.stackHeight = remainingRings.length;
    stack.capHeight = calculateCapHeight(remainingRings);
    stack.controllingPlayer = remainingRings[0];
  }

  // Update elimination counts
  if (ringsToEliminate > 0) {
    newState.totalRingsEliminated += ringsToEliminate;
    newState.board.eliminatedRings[topRingOwner] =
      (newState.board.eliminatedRings[topRingOwner] || 0) + ringsToEliminate;

    const player = newState.players.find((p) => p.playerNumber === topRingOwner);
    if (player) {
      player.eliminatedRings += ringsToEliminate;
    }
  }

  newState.lastMoveAt = new Date();
  return newState;
}
