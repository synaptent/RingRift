import { BoardState, Territory, Position, positionToString } from '../types/game';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from './territoryDetection';
import { getBorderMarkerPositionsForRegion } from './territoryBorders';
import { isStackEligibleForElimination } from './aggregates/EliminationAggregate';

/**
 * Shared territory-processing helpers.
 *
 * This module centralises the **board-level** semantics for disconnected
 * territory regions:
 *
 * - Canonical self-elimination prerequisite (FAQ Q23 / §12.2):
 *   a region is only processable for a player if they control at least one
 *   stack/cap **outside** the region.
 * - Core region application:
 *   - eliminate all stacks (all colours) inside the region;
 *   - collapse region spaces and their marker border to the moving player;
 *   - credit all internal eliminations to the moving player.
 *
 * These helpers are intentionally independent of GameState, turn sequencing,
 * and interaction flows. Backend GameEngine, RuleEngine, and the client
 * sandbox all delegate to this module so that territory geometry and
 * self-elimination semantics have a single source of truth.
 */

/**
 * Minimal context required to process a region on a given board.
 */
export interface TerritoryProcessingContext {
  /** Player who is claiming the region and receiving credited eliminations. */
  player: number;
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
 *
 * All deltas are **per-region**; callers are responsible for updating
 * GameState-level aggregates (players[n].territorySpaces,
 * players[n].eliminatedRings, GameState.totalRingsEliminated, etc.).
 */
export interface TerritoryProcessingOutcome {
  /**
   * Updated board after applying internal eliminations and collapsing region
   * spaces + border markers. The input board is treated as immutable; this
   * field contains a shallow clone with cloned Maps.
   */
  board: BoardState;

  /** The region that was processed (as passed to applyTerritoryRegion). */
  region: Territory;

  /**
   * Border marker positions that were collapsed as part of processing the
   * region, in stable order.
   */
  borderMarkers: Position[];

  /**
   * Per-player eliminated-ring deltas caused by internal eliminations inside
   * the region. Under current rules, all such eliminations are credited to
   * ctx.player, regardless of the colour of the eliminated stacks.
   */
  eliminatedRingsByPlayer: { [player: number]: number };

  /**
   * Per-player territory-space gains caused by collapsing region spaces and
   * border markers. Under current rules, all such gains are credited to
   * ctx.player.
   */
  territoryGainedByPlayer: { [player: number]: number };
}

/**
 * Clone a BoardState for functional-style mutation.
 *
 * This mirrors the cloning used by GameEngine.getGameState / ClientSandboxEngine.
 */
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

/**
 * Self-elimination prerequisite (FAQ Q23 / §12.2 / RR-CANON-R082):
 *
 * A disconnected region is processable for ctx.player **iff** they can pay
 * the mandatory self-elimination cost from **outside** the region.
 *
 * - **Normal territory context:** an eligible cap target outside the region
 *   must exist under RR-CANON-R145.
 * - **Recovery context:** an eligible buried-ring extraction target outside the
 *   region must exist under RR-CANON-R114 (stack need not be controlled).
 *
 * All controlled stacks (including height-1 standalone rings) are eligible
 * cap targets for normal territory processing. Recovery extraction requires
 * a buried ring, so eligible recovery targets necessarily have stack height ≥ 2.
 *
 * This helper performs only the outside-stack check; it assumes that `region`
 * itself is already a disconnected region as reported by the canonical
 * detector.
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
        // Recovery-context territory processing (RR-CANON-R114): the stack
        // need not be controlled by ctx.player; it must contain a buried ring of
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
    }
  }

  // No eligible self-elimination targets for this player outside the region.
  return false;
}

/**
 * Filter a pre-computed list of disconnected regions down to those that are
 * processable for the given player under the self-elimination prerequisite.
 *
 * Unlike {@link getProcessableTerritoryRegions}, this helper does **not**
 * perform region detection itself, making it suitable for callers that want
 * to stub or cache disconnected-region enumeration (e.g. rules-layer tests).
 */
export function filterProcessableTerritoryRegions(
  board: BoardState,
  regions: Territory[],
  ctx: TerritoryProcessingContext
): Territory[] {
  if (regions.length === 0) {
    return [];
  }

  // Canonical guard: disconnected regions must be attributed to a player.
  // A controllingPlayer of 0 indicates a non-canonical detection/recording.
  regions.forEach((region) => {
    if (region.controllingPlayer === 0) {
      throw new Error(
        'Non-canonical territory region: controllingPlayer=0 during territory processing'
      );
    }
  });

  return regions.filter((region) => canProcessTerritoryRegion(board, region, ctx));
}

/**
 * Enumerate disconnected regions that are processable for the given player on
 * the supplied board. This wraps the shared detector +
 * {@link filterProcessableTerritoryRegions} so most callers can use a single
 * entry point.
 */
export function getProcessableTerritoryRegions(
  board: BoardState,
  ctx: TerritoryProcessingContext
): Territory[] {
  const allRegions = findDisconnectedRegionsShared(board);
  const processableRegions = filterProcessableTerritoryRegions(board, allRegions, ctx);
  return processableRegions;
}

/**
 * Apply the **core** consequences of processing a single disconnected region:
 *
 * 1. Eliminate all stacks (all colours) inside the region, crediting all such
 *    eliminations to ctx.player.
 * 2. Collapse all region spaces to ctx.player's colour.
 * 3. Enumerate the marker border for this region and collapse those spaces to
 *    ctx.player as well.
 * 4. Update board.eliminatedRings to reflect the credited eliminations.
 *
 * The mandatory *self-elimination* that follows region processing is **not**
 * handled here; higher-level engines (backend GameEngine, RuleEngine, client
 * sandbox) layer that step on top so it can be expressed as an explicit decision.
 *
 * **Self-Elimination Cost Rules (§12.2):**
 * - Normal territory processing: Player must eliminate an **entire stack cap**
 *   (all consecutive top rings of their colour) from a controlled stack outside
 *   the processed region. All controlled stacks are eligible, including:
 *   (a) Mixed-colour stacks with rings of other colours buried beneath,
 *   (b) Single-colour stacks of height > 1 (all player's rings), AND
 *   (c) Height-1 standalone rings (single-ring stacks).
 * - **Recovery action exception:** When territory processing is triggered by a
 *   recovery action, the cost is only 1 buried ring extraction (bottommost ring
 *   from a chosen stack), not an entire cap.
 *
 * The input `board` is treated as immutable; the returned `board` is a shallow
 * clone with cloned Maps.
 */
export function applyTerritoryRegion(
  board: BoardState,
  region: Territory,
  ctx: TerritoryProcessingContext
): TerritoryProcessingOutcome {
  const nextBoard = cloneBoard(board);

  // 1. Determine border markers for this region using the shared helper.
  const borderMarkers = getBorderMarkerPositionsForRegion(board, region.spaces, {
    mode: 'rust_aligned',
  });

  const regionKeySet = new Set(region.spaces.map((p) => positionToString(p)));

  // 2. Eliminate all stacks inside the region *before* collapsing spaces.
  //    All such eliminations are credited to ctx.player under §12.2.
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
    // Region spaces are interior by construction; they must not remain as
    // markers or stacks once collapsed.
    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, ctx.player);
  }

  // 4. Collapse all border markers to ctx.player's colour. The shared border
  //    helper already omits interior spaces, but we defensively clear stacks
  //    and markers at those positions as well.
  for (const pos of borderMarkers) {
    const key = positionToString(pos);
    if (regionKeySet.has(key)) {
      // Should not happen, but skip rather than double-processing.
      continue;
    }

    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, ctx.player);
  }

  // Territory gain is defined purely by new collapsed spaces for ctx.player:
  // |region.spaces| + |borderMarkers|.
  const territoryGain = region.spaces.length + borderMarkers.length;

  const territoryGainedByPlayer: { [player: number]: number } = {};
  if (territoryGain > 0) {
    territoryGainedByPlayer[ctx.player] = territoryGain;
  }

  // 5. Credit all internal eliminations to ctx.player at the BoardState level.
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
