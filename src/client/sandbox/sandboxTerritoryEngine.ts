import {
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  RegionOrderChoice,
  Territory,
  Move,
  positionToString,
} from '../../shared/types/game';
import {
  findDisconnectedRegionsOnBoard,
  processDisconnectedRegionOnBoard,
  processDisconnectedRegionCoreOnBoard,
} from './sandboxTerritory';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
} from '../../shared/engine/territoryDecisionHelpers';

const TERRITORY_TRACE_DEBUG =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

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

function assertTerritoryEngineMonotonicity(
  context: string,
  before: { collapsedSpaces: number; totalRingsEliminated: number },
  after: { collapsedSpaces: number; totalRingsEliminated: number }
): void {
  const isTestEnv =
    typeof process !== 'undefined' &&
    !!(process as any).env &&
    (process as any).env.NODE_ENV === 'test';

  const errors: string[] = [];

  if (after.collapsedSpaces < before.collapsedSpaces) {
    errors.push(
      `collapsedSpaces decreased in territory engine (${context}): before=${before.collapsedSpaces}, after=${after.collapsedSpaces}`
    );
  }

  if (after.totalRingsEliminated < before.totalRingsEliminated) {
    errors.push(
      `totalRingsEliminated decreased in territory engine (${context}): before=${before.totalRingsEliminated}, after=${after.totalRingsEliminated}`
    );
  }

  if (errors.length === 0) {
    return;
  }

  const message =
    `sandboxTerritoryEngine invariant violation (${context}):` + '\n' + errors.join('\n');

  // eslint-disable-next-line no-console
  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
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
/**
 /**
  * Enumerate canonical territory-processing decision moves for the current
  * player. This mirrors GameEngine.getValidTerritoryProcessingMoves but now
  * delegates region discovery + Q23 gating to the shared
  * {@link enumerateProcessTerritoryRegionMoves} helper so that backend and
  * sandbox observe identical decision sets.
  */
export function getValidTerritoryProcessingMoves(
  gameState: GameState,
  canProcessRegion: (regionSpaces: Position[], playerNumber: number, state: GameState) => boolean
): Move[] {
  const currentPlayer = gameState.currentPlayer;

  // Delegate to the shared helper so region enumeration and
  // self-elimination prerequisites are identical across hosts.
  const rawMoves = enumerateProcessTerritoryRegionMoves(gameState, currentPlayer);

  if (rawMoves.length === 0) {
    return rawMoves;
  }

  // For defensive parity with legacy behaviour, also apply the caller's
  // canProcessRegion predicate as an additional filter. Under normal
  // sandbox flows this is equivalent to the shared helper's gating.
  return rawMoves.filter((move) => {
    if (!move.disconnectedRegions || move.disconnectedRegions.length === 0) {
      return false;
    }
    const region = move.disconnectedRegions[0];
    return canProcessRegion(region.spaces, currentPlayer, gameState);
  });
}
/**
 * Apply a single territory decision move to the game state.
 *
 * This is now a thin adapter over the shared
 * {@link applyProcessTerritoryRegionDecision} and
 * {@link applyEliminateRingsFromStackDecision} helpers so that sandbox
 * behaviour matches the backend GameEngine / RuleEngine exactly for
 * move-driven territory and self-elimination decisions.
 */
export function applyTerritoryDecisionMove(
  gameState: GameState,
  move: Move,
  _canProcessRegion: (regionSpaces: Position[], playerNumber: number, state: GameState) => boolean
): GameState {
  if (move.type !== 'process_territory_region' && move.type !== 'eliminate_rings_from_stack') {
    return gameState;
  }

  if (move.type === 'process_territory_region') {
    const outcome = applyProcessTerritoryRegionDecision(gameState, move);
    return outcome.nextState;
  }

  // move.type === 'eliminate_rings_from_stack'
  const { nextState } = applyEliminateRingsFromStackDecision(gameState, move);
  return nextState;
}

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

    const eligible: Territory[] = disconnected.filter((region) =>
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
        options: eligible.map((r, index) => {
          const representative = r.spaces[0];
          const regionKey = representative
            ? `${representative.x},${representative.y}`
            : `region-${index}`;

          return {
            regionId: String(index),
            size: r.spaces.length,
            representativePosition: representative,
            /**
             * Stable identifier for the canonical 'process_territory_region'
             * Move that would process this region when enumerated via
             * ClientSandboxEngine.getValidTerritoryProcessingMovesForCurrentPlayer.
             * This mirrors the backend RegionOrderChoice.moveId wiring and
             * allows transports/AI to map the choice option directly onto a
             * Move.id in the unified Move model.
             */
            moveId: `process-region-${index}-${regionKey}`,
          };
        }),
      };

      const response = await interactionHandler.requestChoice(choice);
      const selected = (response as PlayerChoiceResponseFor<RegionOrderChoice>).selectedOption;
      const index = parseInt(selected.regionId, 10);
      const selectedRegion = eligible[index] ?? eligible[0];
      regionSpaces = selectedRegion.spaces;

      if (TERRITORY_TRACE_DEBUG) {
        const spaces = regionSpaces || [];
        const containsPos = (x: number, y: number) => spaces.some((p) => p.x === x && p.y === y);

        // eslint-disable-next-line no-console
        console.log('[sandboxTerritoryEngine] selectedRegion', {
          gameId: state.id,
          movingPlayer,
          selectedIndex: index,
          regionSize: spaces.length,
          regionSample: spaces.slice(0, 8).map((p) => `${p.x},${p.y}`),
          contains_3_7: containsPos(3, 7),
          contains_4_0: containsPos(4, 0),
        });
      }
    }

    if (TERRITORY_TRACE_DEBUG && (!interactionHandler || eligible.length === 1)) {
      const spaces = regionSpaces || [];
      const containsPos = (x: number, y: number) => spaces.some((p) => p.x === x && p.y === y);

      // eslint-disable-next-line no-console
      console.log('[sandboxTerritoryEngine] autoSelectedRegion', {
        gameId: state.id,
        movingPlayer,
        regionSize: spaces.length,
        regionSample: spaces.slice(0, 8).map((p) => `${p.x},${p.y}`),
        contains_3_7: containsPos(3, 7),
        contains_4_0: containsPos(4, 0),
      });
    }

    const beforeSnapshot = {
      collapsedSpaces: state.board.collapsedSpaces.size,
      totalRingsEliminated: state.totalRingsEliminated,
    };

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
      totalRingsEliminated: state.totalRingsEliminated + result.totalRingsEliminatedDelta,
    };

    const afterSnapshot = {
      collapsedSpaces: state.board.collapsedSpaces.size,
      totalRingsEliminated: state.totalRingsEliminated,
    };

    assertTerritoryEngineMonotonicity(
      'processDisconnectedRegionsForCurrentPlayerEngine',
      beforeSnapshot,
      afterSnapshot
    );
  }

  return state;
}
