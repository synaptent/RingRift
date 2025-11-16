import {
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  RegionOrderChoice,
  Territory
} from '../../shared/types/game';
import {
  findDisconnectedRegionsOnBoard,
  processDisconnectedRegionOnBoard
} from './sandboxTerritory';

/**
 * Interaction handler abstraction used by the sandbox territory engine.
 * This mirrors the SandboxInteractionHandler shape but avoids importing
 * ClientSandboxEngine directly to keep modules decoupled.
 */
export interface TerritoryInteractionHandler {
  requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>>;
}

/**
 * Pure territory-processing helper for the sandbox engine.
 *
 * Responsibilities:
 * - Detect disconnected regions for the current player.
 * - Filter by self-elimination prerequisite via canProcessRegion.
 * - When multiple regions are eligible and an interaction handler is
 *   present, emit a RegionOrderChoice and respect the selected order.
 * - Apply region processing via processDisconnectedRegionOnBoard.
 *
 * Returns a new GameState with updated board, players, and
 * totalRingsEliminated.
 */
export async function processDisconnectedRegionsForCurrentPlayerEngine(
  gameState: GameState,
  interactionHandler: TerritoryInteractionHandler | null,
  canProcessRegion: (regionSpaces: Position[], playerNumber: number, state: GameState) => boolean
): Promise<GameState> {
  let state = gameState;
  const movingPlayer = state.currentPlayer;

  // Keep processing until no further eligible regions remain.
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const disconnected = findDisconnectedRegionsOnBoard(state.board);
    if (disconnected.length === 0) {
      break;
    }

    const eligible: Territory[] = disconnected.filter(region =>
      canProcessRegion(region.spaces, movingPlayer, state)
    );

    if (eligible.length === 0) {
      break;
    }

    let regionSpaces = eligible[0].spaces;

    if (interactionHandler && eligible.length > 1) {
      const choice: RegionOrderChoice = {
        id: `sandbox-region-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        gameId: state.id,
        playerNumber: movingPlayer,
        type: 'region_order',
        prompt: 'Choose which disconnected region to process first',
        options: eligible.map((r, index) => ({
          regionId: String(index),
          size: r.spaces.length,
          representativePosition: r.spaces[0]
        }))
      };

      const response = await interactionHandler.requestChoice(choice);
      const selected = (response as PlayerChoiceResponseFor<RegionOrderChoice>)
        .selectedOption;
      const index = parseInt(selected.regionId, 10);
      const selectedRegion = eligible[index] ?? eligible[0];
      regionSpaces = selectedRegion.spaces;
    }

    const result = processDisconnectedRegionOnBoard(
      state.board,
      state.players,
      movingPlayer,
      regionSpaces
    );

    state = {
      ...state,
      board: result.board,
      players: result.players,
      totalRingsEliminated: state.totalRingsEliminated + result.totalRingsEliminatedDelta
    };
  }

  return state;
}
