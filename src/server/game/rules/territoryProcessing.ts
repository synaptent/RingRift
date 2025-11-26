/**
 * Legacy backend territory processing module.
 *
 * **IMPORTANT:** This file exists primarily to provide the backend-specific
 * orchestration layer (player interaction, GameState updates) around the
 * canonical shared territory helpers. The actual territory semantics (region
 * detection, collapse logic, border enumeration) are defined in:
 *
 * - {@link ../../../shared/engine/territoryDetection.ts} - Region detection
 * - {@link ../../../shared/engine/territoryProcessing.ts} - Collapse logic
 * - {@link ../../../shared/engine/territoryBorders.ts} - Border markers
 * - {@link ../../../shared/engine/territoryDecisionHelpers.ts} - Decision helpers
 *
 * All territory-processing logic in this file delegates to those shared modules.
 * Do NOT add new territory processing logic here; extend the shared helpers instead.
 *
 * @module server/game/rules/territoryProcessing
 */
import type {
  GameState,
  Territory,
  PlayerChoiceResponseFor,
  RegionOrderChoice,
  TerritoryProcessingContext,
} from '../../../shared/engine';
import {
  positionToString,
  filterProcessableTerritoryRegions,
  applyTerritoryRegion,
} from '../../../shared/engine';
import { BoardManager } from '../BoardManager';
import { PlayerInteractionManager } from '../PlayerInteractionManager';
import {
  eliminatePlayerRingOrCapWithChoice,
  updatePlayerEliminatedRings,
  updatePlayerTerritorySpaces,
} from './lineProcessing';

export interface TerritoryProcessingDeps {
  boardManager: BoardManager;
  interactionManager?: PlayerInteractionManager | undefined;
}

// Debug flag used by parity/trace harnesses to introspect backend territory
// processing behaviour under seeded AI simulations.
const TERRITORY_TRACE_DEBUG =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

/**
 * Process disconnected regions with chain reactions for the current player.
 *
 * This is a direct extraction of GameEngine.processDisconnectedRegions /
 * processOneDisconnectedRegion /
 * canProcessDisconnectedRegion, rewritten in functional style but preserving
 * semantics.
 */
export async function processDisconnectedRegionsForCurrentPlayer(
  gameState: GameState,
  deps: TerritoryProcessingDeps
): Promise<GameState> {
  const { boardManager, interactionManager } = deps;
  const movingPlayer = gameState.currentPlayer;

  // Keep processing until no more disconnections occur
  while (true) {
    const disconnectedRegions = boardManager.findDisconnectedRegions(gameState.board, movingPlayer);

    if (TERRITORY_TRACE_DEBUG) {
      // eslint-disable-next-line no-console
      console.log('[territoryProcessing] disconnectedRegions', {
        gameId: gameState.id,
        movingPlayer,
        regionCount: disconnectedRegions.length,
        regionSizes: disconnectedRegions.map((r) => r.spaces.length),
      });
    }

    if (disconnectedRegions.length === 0) break;

    // Apply the shared self-elimination prerequisite (FAQ Q23 / §12.2):
    // the moving player must control at least one stack outside each
    // region for it to be processable. Delegating this gating to the
    // shared helper keeps backend, sandbox, and rules-layer tests
    // aligned.
    const ctx: TerritoryProcessingContext = { player: movingPlayer };
    const eligibleRegions = filterProcessableTerritoryRegions(
      gameState.board,
      disconnectedRegions,
      ctx
    );

    if (TERRITORY_TRACE_DEBUG) {
      const debugEligible = eligibleRegions.map((region, eligibleIndex) => {
        const spaces = region.spaces || [];
        const containsPos = (x: number, y: number) => spaces.some((p) => p.x === x && p.y === y);
        const originalIndex = disconnectedRegions.indexOf(region);

        return {
          eligibleIndex,
          originalIndex,
          size: spaces.length,
          sample: spaces.slice(0, 8).map(positionToString),
          contains_3_7: containsPos(3, 7),
          contains_4_0: containsPos(4, 0),
        };
      });

      // eslint-disable-next-line no-console
      console.log('[territoryProcessing] eligibleRegions', {
        gameId: gameState.id,
        movingPlayer,
        eligibleCount: eligibleRegions.length,
        eligibleSizes: eligibleRegions.map((r) => r.spaces.length),
        regions: debugEligible,
      });
    }

    if (eligibleRegions.length === 0) {
      // No region can be processed for this player; stop to avoid
      // infinite loops.
      break;
    }

    let region: Territory;

    if (!interactionManager || eligibleRegions.length === 1) {
      // No manager or only one eligible region: process it directly.
      region = eligibleRegions[0];
    } else {
      const choice: RegionOrderChoice = {
        id: generateUUID(
          'region_order',
          gameState.id,
          gameState.history.length,
          eligibleRegions.length
        ),
        gameId: gameState.id,
        playerNumber: movingPlayer,
        type: 'region_order',
        prompt: 'Choose which disconnected region to process first',
        options: eligibleRegions.map((r, index) => {
          const representative = r.spaces[0];
          const regionKey = representative ? positionToString(representative) : `region-${index}`;

          return {
            regionId: String(index),
            size: r.spaces.length,
            representativePosition: representative,
            /**
             * Stable identifier for the canonical 'process_territory_region'
             * Move that would process this region when enumerated via
             * advanced-phase helpers (RuleEngine.getValidMoves /
             * GameEngine.getValidMoves in the territory_processing phase).
             * This lets transports/AI map this choice option directly onto
             * a Move.id.
             */
            moveId: `process-region-${index}-${regionKey}`,
          };
        }),
      };

      const response: PlayerChoiceResponseFor<RegionOrderChoice> =
        await interactionManager.requestChoice(choice);
      const selected = response.selectedOption;
      const index = parseInt(selected.regionId, 10);
      region = eligibleRegions[index] ?? eligibleRegions[0];
    }

    if (TERRITORY_TRACE_DEBUG) {
      const spaces = region.spaces || [];
      const containsPos = (x: number, y: number) => spaces.some((p) => p.x === x && p.y === y);

      // eslint-disable-next-line no-console
      console.log('[territoryProcessing] processingRegion', {
        gameId: gameState.id,
        movingPlayer,
        regionSize: spaces.length,
        regionSample: spaces.slice(0, 8).map(positionToString),
        contains_3_7: containsPos(3, 7),
        contains_4_0: containsPos(4, 0),
      });
    }

    gameState = await processOneDisconnectedRegion(gameState, region, movingPlayer, deps);
  }

  return gameState;
}

/**
 * Self-elimination prerequisite: player must have at least one stack
 * outside the disconnected region.
 *
 * @deprecated This internal helper is DEAD CODE and is not called anywhere.
 * Use {@link filterProcessableTerritoryRegions} from
 * `src/shared/engine/territoryProcessing.ts` instead, which provides identical
 * semantics and is the canonical source of truth for all engines.
 *
 * This function is retained temporarily for reference but will be removed
 * in a future cleanup pass.
 */
function canProcessDisconnectedRegion(
  gameState: GameState,
  region: Territory,
  player: number,
  deps: TerritoryProcessingDeps
): boolean {
  const { boardManager } = deps;
  const regionPositionSet = new Set(region.spaces.map((pos) => positionToString(pos)));
  const playerStacks = boardManager.getPlayerStacks(gameState.board, player);

  if (TERRITORY_TRACE_DEBUG) {
    const stackKeys = playerStacks.map((s) => positionToString(s.position));
    const allBoardStackKeys = Array.from(gameState.board.stacks.keys());
    // eslint-disable-next-line no-console
    console.log('[territoryProcessing.canProcessDisconnectedRegion]', {
      gameId: gameState.id,
      movingPlayer: player,
      regionSize: region.spaces.length,
      regionSample: region.spaces.slice(0, 8).map(positionToString),
      playerStackCount: playerStacks.length,
      playerStackPositions: stackKeys,
      boardStackCount: allBoardStackKeys.length,
      boardStackKeysSample: allBoardStackKeys.slice(0, 16),
    });
  }

  for (const stack of playerStacks) {
    const stackPosKey = positionToString(stack.position);
    if (!regionPositionSet.has(stackPosKey)) {
      // Found a stack outside the region
      return true;
    }
  }

  // No stacks outside the region - cannot process
  return false;
}

/**
 * Process a single disconnected region.
 *
 * Rule Reference: Section 12.2 - Processing steps
 */
async function processOneDisconnectedRegion(
  gameState: GameState,
  region: Territory,
  movingPlayer: number,
  deps: TerritoryProcessingDeps
): Promise<GameState> {
  // Delegate the geometric core (internal eliminations + region/border
  // collapse) to the shared engine helper so that backend, sandbox, and
  // rules-layer tests share a single source of truth for territory
  // semantics.
  const ctx: TerritoryProcessingContext = { player: movingPlayer };
  const outcome = applyTerritoryRegion(gameState.board, region, ctx);

  // Replace the board with the processed clone from the shared helper.
  gameState = {
    ...gameState,
    board: outcome.board,
  };

  // Apply per-player territory gain at the GameState level. Under current
  // rules all territory gain from disconnected regions is credited to the
  // moving player.
  const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
  if (territoryGain > 0) {
    gameState = updatePlayerTerritorySpaces(gameState, movingPlayer, territoryGain);
  }

  // Apply internal elimination deltas to GameState.totalRingsEliminated and
  // the moving player's eliminatedRings counter. The BoardState-level
  // bookkeeping (board.eliminatedRings) has already been updated by the
  // shared helper.
  const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
  if (internalElims > 0) {
    gameState.totalRingsEliminated += internalElims;
    gameState = updatePlayerEliminatedRings(gameState, movingPlayer, internalElims);
  }

  // 6. Mandatory self-elimination (one ring or cap from moving player).
  // This remains a GameState-level concern and is intentionally layered on
  // top of the shared board-level helper.
  gameState = await eliminatePlayerRingOrCapWithChoice(gameState, movingPlayer, deps);

  return gameState;
}

// Local deterministic identifier helper for territory-related choices.
// This deliberately avoids any RNG so that core rules behaviour remains
// fully deterministic (RR‑CANON R190). Callers pass structured context
// (game id, history length, candidate count, etc.) so IDs remain unique
// and stable for parity/diagnostic tooling.
function generateUUID(...parts: Array<string | number | undefined>): string {
  return parts
    .filter((part) => part !== undefined)
    .map((part) => String(part))
    .join('|');
}
