import {
  GameState,
  Territory,
  Position,
  positionToString,
  PlayerChoiceResponseFor,
  RegionOrderChoice
} from '../../../shared/types/game';
import { BoardManager } from '../BoardManager';
import { PlayerInteractionManager } from '../PlayerInteractionManager';
import {
  eliminatePlayerRingOrCapWithChoice,
  updatePlayerEliminatedRings,
  updatePlayerTerritorySpaces
} from './lineProcessing';

export interface TerritoryProcessingDeps {
  boardManager: BoardManager;
  interactionManager?: PlayerInteractionManager | undefined;
}

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
    const disconnectedRegions = boardManager.findDisconnectedRegions(
      gameState.board,
      movingPlayer
    );

    if (disconnectedRegions.length === 0) break;

    // Filter to regions that satisfy the self-elimination prerequisite
    // for the moving player.
    const eligibleRegions = disconnectedRegions.filter(region =>
      canProcessDisconnectedRegion(gameState, region, movingPlayer, deps)
    );

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
        options: eligibleRegions.map((r, index) => ({
          regionId: String(index),
          size: r.spaces.length,
          representativePosition: r.spaces[0]
        }))
      };

      const response: PlayerChoiceResponseFor<RegionOrderChoice> =
        await interactionManager.requestChoice(choice);
      const selected = response.selectedOption;
      const index = parseInt(selected.regionId, 10);
      region = eligibleRegions[index] ?? eligibleRegions[0];
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
  const regionPositionSet = new Set(region.spaces.map(pos => positionToString(pos)));
  const playerStacks = boardManager.getPlayerStacks(gameState.board, player);

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
  const borderMarkers = boardManager.getBorderMarkerPositions(
    region.spaces,
    gameState.board
  );

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
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
