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
} from './sandboxTerritory';
import { forceEliminateCapOnBoard } from './sandboxElimination';

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
 * Enumerate canonical territory-processing decision moves for the current
 * player. This mirrors GameEngine.getValidTerritoryProcessingMoves.
 */
export function getValidTerritoryProcessingMoves(
  gameState: GameState,
  canProcessRegion: (regionSpaces: Position[], playerNumber: number, state: GameState) => boolean
): Move[] {
  const moves: Move[] = [];
  const currentPlayer = gameState.currentPlayer;

  const disconnectedRegions = findDisconnectedRegionsOnBoard(gameState.board);

  if (disconnectedRegions.length === 0) {
    return moves;
  }

  const eligibleRegions = disconnectedRegions.filter((region) =>
    canProcessRegion(region.spaces, currentPlayer, gameState)
  );

  if (eligibleRegions.length === 0) {
    return moves;
  }

  // One process_territory_region move per eligible disconnected region
  eligibleRegions.forEach((region, index) => {
    const representative = region.spaces[0];
    const regionKey = representative ? positionToString(representative) : `region-${index}`;
    moves.push({
      id: `process-region-${index}-${regionKey}`,
      type: 'process_territory_region',
      player: currentPlayer,
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.moveHistory.length + 1,
    } as Move);
  });

  return moves;
}

/**
 * Apply a single territory decision move to the game state.
 */
export function applyTerritoryDecisionMove(
  gameState: GameState,
  move: Move,
  canProcessRegion: (regionSpaces: Position[], playerNumber: number, state: GameState) => boolean
): GameState {
  if (move.type !== 'process_territory_region' && move.type !== 'eliminate_rings_from_stack') {
    return gameState;
  }

  let state = gameState;
  const movingPlayer = move.player;

  if (move.type === 'process_territory_region') {
    if (!move.disconnectedRegions || move.disconnectedRegions.length === 0) {
      return state;
    }

    const targetRegion = move.disconnectedRegions[0];

    if (!canProcessRegion(targetRegion.spaces, movingPlayer, state)) {
      return state;
    }

    const result = processDisconnectedRegionOnBoard(
      state.board,
      state.players,
      movingPlayer,
      targetRegion.spaces
    );

    state = {
      ...state,
      board: result.board,
      players: result.players,
      totalRingsEliminated: state.totalRingsEliminated + result.totalRingsEliminatedDelta,
    };
  } else if (move.type === 'eliminate_rings_from_stack') {
    if (!move.to) {
      return state;
    }

    const stackKey = positionToString(move.to);
    const stack = state.board.stacks.get(stackKey);

    if (!stack || stack.controllingPlayer !== movingPlayer) {
      return state;
    }

    const stacks = [stack];
    const elimResult = forceEliminateCapOnBoard(state.board, state.players, movingPlayer, stacks);

    state = {
      ...state,
      board: elimResult.board,
      players: elimResult.players,
      totalRingsEliminated: state.totalRingsEliminated + elimResult.totalRingsEliminatedDelta,
    };
  }

  return state;
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
