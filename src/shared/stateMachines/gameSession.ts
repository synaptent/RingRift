import type { GamePhase, GameResult, GameState, GameStatus } from '../types/game';

/**
 * Explicit session-level view of a game, derived from the canonical
 * {@link GameState}. This does not replace GameState as the source of
 * truth – it provides a smaller, intent-focused lens for orchestrating
 * timers, AI turns, and WebSocket flows.
 */

export type LiveGameStatus = Extract<GameStatus, 'waiting' | 'active' | 'paused'>;
export type TerminalGameStatus = Extract<GameStatus, 'completed' | 'finished' | 'abandoned'>;

export interface WaitingForPlayersSession {
  kind: 'waiting_for_players';
  gameId: string;
  status: Extract<GameStatus, 'waiting'>;
}

export interface ActiveTurnSession {
  kind: 'active_turn';
  gameId: string;
  status: Extract<GameStatus, 'active' | 'paused'>;
  currentPlayer: number;
  phase: GamePhase;
}

export interface CompletedSession {
  kind: 'completed';
  gameId: string;
  status: Extract<GameStatus, 'completed' | 'finished'>;
  /** Optional, when a structured GameResult is available. */
  result?: GameResult | undefined;
}

export interface AbandonedSession {
  kind: 'abandoned';
  gameId: string;
  status: Extract<GameStatus, 'abandoned'>;
  /** Optional snapshot of scores at abandonment time. */
  result?: GameResult | undefined;
}

export type GameSessionStatus =
  | WaitingForPlayersSession
  | ActiveTurnSession
  | CompletedSession
  | AbandonedSession;

/**
 * Lightweight, pure derivation of a {@link GameSessionStatus} from the current
 * {@link GameState}. This helper is intentionally side-effect free; callers
 * are responsible for deciding when to recompute or persist the derived
 * status.
 */
export function deriveGameSessionStatus(state: GameState, result?: GameResult): GameSessionStatus {
  const base = {
    gameId: state.id,
  } as const;

  const status = state.gameStatus;

  if (status === 'waiting') {
    return {
      kind: 'waiting_for_players',
      status,
      ...base,
    };
  }

  if (status === 'active' || status === 'paused') {
    return {
      kind: 'active_turn',
      status,
      currentPlayer: state.currentPlayer,
      phase: state.currentPhase,
      ...base,
    };
  }

  if (status === 'abandoned') {
    return {
      kind: 'abandoned',
      status,
      result,
      ...base,
    };
  }

  // Treat any other terminal status as a completed session. This covers
  // both the legacy 'finished' value and the newer 'completed' value.
  return {
    kind: 'completed',
    status: (status as Extract<GameStatus, 'completed' | 'finished'>) ?? 'completed',
    result,
    ...base,
  };
}
