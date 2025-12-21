/**
 * GameFacade - Unified abstraction for game host implementations
 *
 * This facade provides a common interface for both backend-connected games
 * (BackendGameHost) and local sandbox games (SandboxGameHost). By abstracting
 * the underlying game source, shared UI components can render consistently
 * regardless of whether the game is server-authoritative or client-local.
 *
 * The facade exposes:
 * - Read-only game state and derived computed values
 * - Move submission callbacks
 * - Player choice/decision handling
 * - Connection/mode information
 *
 * Implementations:
 * - `BackendGameFacade`: Wraps GameContext hooks for server-backed games
 * - `SandboxGameFacade`: Wraps ClientSandboxEngine for local games
 *
 * @module facades/GameFacade
 */

import type {
  BoardType,
  GameState,
  GameResult,
  Move,
  PlayerChoice,
  Position,
} from '../../shared/types/game';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Partial move for submission (excludes auto-generated fields).
 */
export interface PartialMove {
  type: Move['type'];
  from?: Position;
  to: Position;
  placementCount?: number;
  placedOnStack?: boolean;
  captureTarget?: Position;
}

/**
 * Connection status for the game facade.
 */
export type FacadeConnectionStatus =
  | 'connected'
  | 'connecting'
  | 'reconnecting'
  | 'disconnected'
  | 'local-only';

/**
 * Game mode indicator.
 */
export type GameFacadeMode = 'backend' | 'sandbox' | 'spectator';

/**
 * Decision phase state exposed by the facade.
 */
export interface FacadeDecisionState {
  /** Currently pending choice (if any) */
  pendingChoice: PlayerChoice | null;
  /** Deadline timestamp for the choice (ms since epoch, if applicable) */
  choiceDeadline: number | null;
  /** Remaining time in ms (client-reconciled) */
  choiceTimeRemainingMs: number | null;
  /** Whether the time is server-capped (backend only) */
  isServerCapped?: boolean;
}

/**
 * Player information for UI display.
 */
export interface FacadePlayerInfo {
  playerNumber: number;
  username: string;
  type: 'human' | 'ai';
  ringsInHand: number;
  eliminatedRings: number;
  territorySpaces: number;
  isCurrentPlayer: boolean;
  isLocalUser: boolean;
}

/**
 * Chain capture state for visualization.
 */
export interface FacadeChainCaptureState {
  /** Path of positions in the current chain capture */
  path: Position[];
  /** Whether chain capture must continue */
  mustContinue: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN FACADE INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * GameFacade - Common interface for game state and actions.
 *
 * This interface abstracts the differences between backend-connected games
 * and local sandbox games, allowing shared UI components to work with either.
 */
export interface GameFacade {
  // ─────────────────────────────────────────────────────────────────────────
  // Core Game State
  // ─────────────────────────────────────────────────────────────────────────

  /** Current game state (null if not loaded) */
  readonly gameState: GameState | null;

  /** Valid moves for the current player */
  readonly validMoves: Move[];

  /** Victory/end result (null if game is active) */
  readonly victoryState: GameResult | null;

  /** Structured game end explanation (if game ended) */
  readonly gameEndExplanation: GameEndExplanation | null;

  // ─────────────────────────────────────────────────────────────────────────
  // Mode & Connection
  // ─────────────────────────────────────────────────────────────────────────

  /** Game mode (backend, sandbox, or spectator) */
  readonly mode: GameFacadeMode;

  /** Connection status */
  readonly connectionStatus: FacadeConnectionStatus;

  /** Whether the local user is a player in this game */
  readonly isPlayer: boolean;

  /** Whether it's the local user's turn */
  readonly isMyTurn: boolean;

  /** Board type for this game */
  readonly boardType: BoardType;

  /** Current user's ID (if authenticated) */
  readonly currentUserId: string | undefined;

  // ─────────────────────────────────────────────────────────────────────────
  // Decision/Choice State
  // ─────────────────────────────────────────────────────────────────────────

  /** Decision phase state (pending choices, timeouts) */
  readonly decisionState: FacadeDecisionState;

  // ─────────────────────────────────────────────────────────────────────────
  // Derived State (convenience)
  // ─────────────────────────────────────────────────────────────────────────

  /** Chain capture path (if in chain capture phase) */
  readonly chainCaptureState: FacadeChainCaptureState | null;

  /** Must-move-from position (if all valid moves share an origin) */
  readonly mustMoveFrom: Position | undefined;

  /** Player information for HUD */
  readonly players: FacadePlayerInfo[];

  // ─────────────────────────────────────────────────────────────────────────
  // Actions
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Submit a move to the game.
   *
   * For backend games, this sends the move to the server.
   * For sandbox games, this applies the move locally via the engine.
   */
  submitMove(move: PartialMove): void;

  /**
   * Respond to a player choice/decision.
   *
   * @param choice The choice being responded to
   * @param selectedOption The selected option from choice.options
   */
  respondToChoice<T extends PlayerChoice>(choice: T, selectedOption: T['options'][number]): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Optional: Evaluation (for analysis panels)
  // ─────────────────────────────────────────────────────────────────────────

  /** AI evaluation history (if available) */
  readonly evaluationHistory?: PositionEvaluationPayload['data'][];
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Extract chain capture path from move history.
 *
 * This is a common utility used by both backend and sandbox facades to
 * derive the chain capture visualization path from game state.
 */
export function extractChainCapturePath(gameState: GameState): Position[] | undefined {
  if (gameState.currentPhase !== 'chain_capture') {
    return undefined;
  }

  const moveHistory = gameState.moveHistory;
  if (!moveHistory || moveHistory.length === 0) {
    return undefined;
  }

  const currentPlayer = gameState.currentPlayer;
  const path: Position[] = [];

  // Walk backwards to find all chain capture moves by the current player
  for (let i = moveHistory.length - 1; i >= 0; i--) {
    const move = moveHistory[i];
    if (!move) continue;

    // Stop if we hit a move by a different player or a non-capture move
    if (
      move.player !== currentPlayer ||
      (move.type !== 'overtaking_capture' && move.type !== 'continue_capture_segment')
    ) {
      break;
    }

    // Add the landing position to the front of the path
    if (move.to) {
      path.unshift(move.to);
    }

    // If this is the first capture in the chain, add the starting position
    if (move.type === 'overtaking_capture' && move.from) {
      path.unshift(move.from);
    }
  }

  // Need at least 2 positions to show a path
  return path.length >= 2 ? path : undefined;
}

/**
 * Derive must-move-from position from valid moves.
 *
 * When all movement/capture moves originate from the same stack,
 * that stack is treated as the forced origin for highlighting.
 */
export function deriveMustMoveFrom(
  validMoves: Move[],
  gameState: GameState | null
): Position | undefined {
  if (!validMoves || validMoves.length === 0 || !gameState) {
    return undefined;
  }

  if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
    return undefined;
  }

  const origins = validMoves
    .filter((m) => m.from && (m.type === 'move_stack' || m.type === 'overtaking_capture'))
    .map((m) => m.from as Position);

  if (origins.length === 0) {
    return undefined;
  }

  const first = origins[0];
  const allSame = origins.every((p) => p.x === first.x && p.y === first.y && p.z === first.z);

  return allSame ? first : undefined;
}

/**
 * Check if the facade is in a state where moves can be submitted.
 */
export function canSubmitMove(facade: GameFacade): boolean {
  if (!facade.gameState || facade.gameState.gameStatus !== 'active') {
    return false;
  }

  if (!facade.isPlayer) {
    return false;
  }

  if (facade.connectionStatus === 'disconnected') {
    return false;
  }

  return true;
}

/**
 * Check if the facade is in a state where interactions are allowed.
 */
export function canInteract(facade: GameFacade): boolean {
  if (!facade.gameState) {
    return false;
  }

  if (facade.mode === 'spectator') {
    return false;
  }

  if (facade.connectionStatus === 'disconnected' && facade.mode === 'backend') {
    return false;
  }

  return true;
}
