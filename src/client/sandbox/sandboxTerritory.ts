/**
 * @fileoverview Sandbox Territory Helpers - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It provides territory detection and processing for sandbox/offline games.
 *
 * Canonical SSoT:
 * - Region detection: `src/shared/engine/territoryDetection.ts`
 * - Collapse logic: `src/shared/engine/territoryProcessing.ts`
 * - Border markers: `src/shared/engine/territoryBorders.ts`
 * - Decision helpers: `src/shared/engine/territoryDecisionHelpers.ts`
 * - Territory aggregate: `src/shared/engine/aggregates/TerritoryAggregate.ts`
 *
 * This adapter:
 * - Delegates region detection to `findDisconnectedRegions()`
 * - Delegates border enumeration to `getBorderMarkerPositionsForRegion()`
 * - Delegates region processing to `applyTerritoryRegion()`
 * - Contains NO standalone region-geometry implementation
 *
 * DO NOT add territory processing logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 * @module client/sandbox/sandboxTerritory
 */
import type {
  BoardState,
  Player,
  Position,
  Territory,
  RingStack,
  TerritoryProcessingContext,
} from '../../shared/engine';
import {
  findDisconnectedRegions as findDisconnectedRegionsShared,
  getBorderMarkerPositionsForRegion as getSharedBorderMarkers,
  applyTerritoryRegion,
} from '../../shared/engine';
import { forceEliminateCapOnBoard, ForcedEliminationResult } from './sandboxElimination';

/**
 * Compute all disconnected regions on the board.
 *
 * Sandbox now delegates to the shared detector in
 * src/shared/engine/territoryDetection.ts so that region geometry is
 * defined in exactly one place for backend and client engines.
 */
export function findDisconnectedRegionsOnBoard(board: BoardState): Territory[] {
  return findDisconnectedRegionsShared(board);
}

/**
 * Identify all marker positions that form the border around a disconnected
 * region. This delegates to the shared helper in
 * src/shared/engine/territoryBorders.ts so backend and sandbox cannot
 * drift.
 */
export function getBorderMarkerPositionsForRegion(
  board: BoardState,
  regionSpaces: Position[]
): Position[] {
  return getSharedBorderMarkers(board, regionSpaces, { mode: 'rust_aligned' });
}

/**
 * Process a single disconnected region for the moving player, applying the
 * collapse and elimination rules described in Section 6 of the compact
 * rules. This function is pure with respect to GameState: callers must
 * update totalRingsEliminated themselves using the returned delta.
 */
export function processDisconnectedRegionCoreOnBoard(
  board: BoardState,
  players: Player[],
  movingPlayer: number,
  regionSpaces: Position[]
): { board: BoardState; players: Player[]; totalRingsEliminatedDelta: number } {
  // Delegate the geometric core (internal eliminations + region/border
  // collapse) to the shared engine helper so that backend, sandbox, and
  // rules-layer tests all share a single source of truth for territory
  // semantics.
  const region: Territory = {
    spaces: regionSpaces,
    controllingPlayer: movingPlayer,
    isDisconnected: true,
  };

  const ctx: TerritoryProcessingContext = { player: movingPlayer };
  const outcome = applyTerritoryRegion(board, region, ctx);

  let nextPlayers: Player[] = players.map((p) => ({ ...p }));

  // Update territorySpaces for moving player based on the shared outcome.
  const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
  if (territoryGain > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === movingPlayer
        ? { ...p, territorySpaces: p.territorySpaces + territoryGain }
        : p
    );
  }

  // Credit all internal eliminations to the moving player at the Player[]
  // level. BoardState-level eliminatedRings has already been updated by
  // applyTerritoryRegion.
  const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
  if (internalElims > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === movingPlayer
        ? { ...p, eliminatedRings: p.eliminatedRings + internalElims }
        : p
    );
  }

  return {
    board: outcome.board,
    players: nextPlayers,
    totalRingsEliminatedDelta: internalElims,
  };
}

export function processDisconnectedRegionOnBoard(
  board: BoardState,
  players: Player[],
  movingPlayer: number,
  regionSpaces: Position[]
): { board: BoardState; players: Player[]; totalRingsEliminatedDelta: number } {
  // Legacy / non-move-driven helper: apply the core geometric consequences of
  // processing a disconnected region and then immediately perform the mandatory
  // self-elimination using forceEliminateCapOnBoard. This preserves the
  // original sandbox semantics used by processDisconnectedRegionsForCurrentPlayerEngine.
  const coreResult = processDisconnectedRegionCoreOnBoard(
    board,
    players,
    movingPlayer,
    regionSpaces
  );

  const nextBoard = coreResult.board;
  const nextPlayers = coreResult.players;

  // 6. Mandatory self-elimination: eliminate one cap from a moving-player stack
  // outside the region using the shared forceEliminateCapOnBoard helper.
  const movingStacks: RingStack[] = [];
  for (const stack of nextBoard.stacks.values()) {
    if (stack.controllingPlayer === movingPlayer) {
      movingStacks.push(stack);
    }
  }

  const elimResult: ForcedEliminationResult = forceEliminateCapOnBoard(
    nextBoard,
    nextPlayers,
    movingPlayer,
    movingStacks
  );

  const totalDelta = coreResult.totalRingsEliminatedDelta + elimResult.totalRingsEliminatedDelta;

  return {
    board: elimResult.board,
    players: elimResult.players,
    totalRingsEliminatedDelta: totalDelta,
  };
}
