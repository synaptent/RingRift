import {
  GameState,
  Territory,
  positionToString,
  PlayerChoiceResponseFor,
  RegionOrderChoice,
} from '../../../shared/types/game';
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

    // Filter to regions that satisfy the self-elimination prerequisite
    // for the moving player.
    const eligibleRegions = disconnectedRegions.filter((region) =>
      canProcessDisconnectedRegion(gameState, region, movingPlayer, deps)
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
        id: generateUUID(),
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
  const { boardManager } = deps;

  // 1. Get border markers to collapse
  const borderMarkers = boardManager.getBorderMarkerPositions(region.spaces, gameState.board);

  // 2. Eliminate all rings within the region (all colors) BEFORE
  //    collapsing spaces.
  let totalRingsEliminated = 0;
  for (const pos of region.spaces) {
    const stack = boardManager.getStack(pos, gameState.board);
    if (stack) {
      totalRingsEliminated += stack.stackHeight;
      boardManager.removeStack(pos, gameState.board);
    }
  }

  // 3. Collapse all spaces in the region to the moving player's color
  for (const pos of region.spaces) {
    boardManager.setCollapsedSpace(pos, movingPlayer, gameState.board);
  }

  // 4. Collapse all border markers to the moving player's color
  for (const pos of borderMarkers) {
    boardManager.setCollapsedSpace(pos, movingPlayer, gameState.board);
  }

  // Update player's territory count (region spaces + border markers)
  const totalTerritoryGained = region.spaces.length + borderMarkers.length;
  gameState = updatePlayerTerritorySpaces(gameState, movingPlayer, totalTerritoryGained);

  // 5. Update elimination counts - ALL eliminated rings count toward moving player
  gameState.totalRingsEliminated += totalRingsEliminated;
  if (!gameState.board.eliminatedRings[movingPlayer]) {
    gameState.board.eliminatedRings[movingPlayer] = 0;
  }
  gameState.board.eliminatedRings[movingPlayer] += totalRingsEliminated;

  gameState = updatePlayerEliminatedRings(gameState, movingPlayer, totalRingsEliminated);

  // 6. Mandatory self-elimination (one ring or cap from moving player)
  gameState = await eliminatePlayerRingOrCapWithChoice(gameState, movingPlayer, deps);

  return gameState;
}

// Local UUID generator mirroring GameEngine.generateUUID
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
