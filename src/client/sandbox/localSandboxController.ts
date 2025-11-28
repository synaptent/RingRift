import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  GamePhase,
  GameState,
  Player,
  Position,
  RingStack,
  positionToString,
} from '../../shared/types/game';
import { calculateCapHeight, getPathPositions } from '../../shared/engine/core';

/**
 * DIAGNOSTICS-ONLY (legacy) – local sandbox harness.
 *
 * This module implements a minimal, browser-safe local sandbox rules subset
 * independent of the shared turn orchestrator. It exists solely for
 * diagnostics, experimentation, and Phase C migration work; it is **not**
 * used by the `/sandbox` route or [`GamePage`](../pages/GamePage.tsx:165) in
 * normal operation.
 *
 * Canonical sandbox rules behaviour is provided by:
 * - [`ClientSandboxEngine`](./ClientSandboxEngine.ts:131) as the sandbox host.
 * - [`SandboxOrchestratorAdapter`](./SandboxOrchestratorAdapter.ts:162) +
 *   `processTurnAsync` in the shared orchestrator as the lifecycle SSOT.
 *
 * See [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](../../../docs/ORCHESTRATOR_ROLLOUT_PLAN.md:302)
 * Phase B/Phase C for the orchestrator-first rollout and legacy shutdown plan.
 *
 * This harness provides just enough structure for diagnostics code to manage a
 * local board and per-player ring placement/movement in a way that is
 * compatible with the shared `GameState`/`BoardState` types, without importing
 * any server-only modules. It must not be reintroduced as a production rules
 * host for `/sandbox`.
 */

export interface LocalSandboxConfig {
  boardType: BoardType;
  numPlayers: number;
}

export interface LocalSandboxState {
  board: BoardState;
  players: Player[];
  currentPlayer: number;
  currentPhase: GamePhase;
  /**
   * Optional selection used during the movement phase to remember the
   * currently selected stack. This is confined to the sandbox and does
   * not affect the shared GameState shape.
   */
  selectedStack?: Position;
}

/**
 * Create an empty BoardState for the given boardType, mirroring the
 * server-side BoardManager.createBoard shape but without importing the
 * server module into the client bundle.
 */
export function createEmptyBoard(boardType: BoardType): BoardState {
  const config = BOARD_CONFIGS[boardType];
  return {
    stacks: new Map<string, RingStack>(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: config.size,
    type: boardType,
  };
}

/**
 * Initialize a minimal local sandbox state. Players are given
 * placeholder identities; the sandbox currently focuses on board
 * behaviour rather than authentication.
 */
export function createInitialLocalSandboxState(config: LocalSandboxConfig): LocalSandboxState {
  const players: Player[] = Array.from({ length: config.numPlayers }, (_, idx) => {
    const playerNumber = idx + 1;
    return {
      id: `local-${playerNumber}`,
      username: `Player ${playerNumber}`,
      type: 'human',
      playerNumber,
      isReady: true,
      timeRemaining: 0,
      aiDifficulty: undefined,
      ringsInHand: BOARD_CONFIGS[config.boardType].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    };
  });

  return {
    board: createEmptyBoard(config.boardType),
    players,
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    selectedStack: undefined,
  };
}

/**
 * Very small subset of GameState projected for the sandbox UI. This
 * allows us to reuse some HUD components later if needed without
 * claiming full rules fidelity.
 */
export function toSandboxGameState(
  state: LocalSandboxState
): Pick<GameState, 'board' | 'players' | 'currentPlayer' | 'currentPhase'> {
  return {
    board: state.board,
    players: state.players,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
  } as Pick<GameState, 'board' | 'players' | 'currentPlayer' | 'currentPhase'>;
}

/**
 * Handle a cell click in the local sandbox. For now, this implements
 * a very small, explicitly experimental rule subset:
 *
 * - In the ring_placement phase, clicking an empty cell places a
 *   single ring for the current player (if they still have rings in
 *   hand) and advances to the next player.
 * - Other phases are currently treated as no-ops; future work will
 *   extend this function to call into a shared GameEngine wrapper.
 */
export function handleLocalSandboxCellClick(
  state: LocalSandboxState,
  position: Position
): LocalSandboxState {
  if (state.currentPhase === 'ring_placement') {
    const key = positionToString(position);
    const existingStack = state.board.stacks.get(key);
    if (existingStack) {
      // Do not stack in the simplistic placement phase; future work can
      // evolve this into full placement rules.
      return state;
    }

    const player = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (!player || player.ringsInHand <= 0) {
      // No rings left to place for this player; ignore the click for now.
      return state;
    }

    const rings = [state.currentPlayer];
    const newStack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: state.currentPlayer,
    };

    const nextStacks = new Map(state.board.stacks);
    nextStacks.set(key, newStack);

    const nextBoard: BoardState = {
      ...state.board,
      stacks: nextStacks,
    };

    const updatedPlayers = state.players.map((p) =>
      p.playerNumber === state.currentPlayer
        ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - 1) }
        : p
    );

    const nextPlayerIndex =
      updatedPlayers.findIndex((p) => p.playerNumber === state.currentPlayer) + 1;
    const nextPlayer =
      nextPlayerIndex < updatedPlayers.length
        ? updatedPlayers[nextPlayerIndex].playerNumber
        : updatedPlayers[0].playerNumber;

    const allRingsExhausted = updatedPlayers.every((p) => p.ringsInHand <= 0);

    return {
      board: nextBoard,
      players: updatedPlayers,
      currentPlayer: nextPlayer,
      currentPhase: allRingsExhausted ? 'movement' : 'ring_placement',
      selectedStack: undefined,
    };
  }

  if (state.currentPhase === 'movement') {
    const key = positionToString(position);
    const stackAtPos = state.board.stacks.get(key);

    // If clicking on a stack belonging to the current player, (re)select it.
    if (stackAtPos && stackAtPos.controllingPlayer === state.currentPlayer) {
      return {
        ...state,
        selectedStack: position,
      };
    }

    // If we don't have a selected stack yet, nothing to do.
    if (!state.selectedStack) {
      return state;
    }

    const fromKey = positionToString(state.selectedStack);
    const movingStack = state.board.stacks.get(fromKey);
    if (!movingStack || movingStack.controllingPlayer !== state.currentPlayer) {
      return state;
    }

    // Disallow moves onto occupied stacks or collapsed spaces in this
    // minimal sandbox movement phase.
    if (stackAtPos || state.board.collapsedSpaces.has(key)) {
      return state;
    }

    // Use the shared path helper to ensure the path is unobstructed by
    // stacks or collapsed spaces. This is intentionally simpler than
    // the full RuleEngine but grounded in the same geometry.
    const path = getPathPositions(state.selectedStack, position).slice(1, -1);
    for (const pos of path) {
      const pathKey = positionToString(pos);
      if (state.board.collapsedSpaces.has(pathKey) || state.board.stacks.has(pathKey)) {
        return state;
      }
    }

    const nextStacks = new Map(state.board.stacks);
    nextStacks.delete(fromKey);
    nextStacks.set(key, {
      ...movingStack,
      position,
    });

    const nextBoard: BoardState = {
      ...state.board,
      stacks: nextStacks,
    };

    const nextPlayerIndex =
      state.players.findIndex((p) => p.playerNumber === state.currentPlayer) + 1;
    const nextPlayer =
      nextPlayerIndex < state.players.length
        ? state.players[nextPlayerIndex].playerNumber
        : state.players[0].playerNumber;

    return {
      ...state,
      board: nextBoard,
      currentPlayer: nextPlayer,
      selectedStack: undefined,
    };
  }

  // Other phases are currently treated as no-ops; future work will
  // extend this function to call into a shared GameEngine wrapper.
  return state;
}
