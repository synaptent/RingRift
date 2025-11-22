import { GameState, GamePhase, GameStatus } from '../types';

export function mutateTurnChange(state: GameState): GameState {
  const newState = {
    ...state,
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    currentPlayer: number;
    currentPhase: GamePhase;
    gameStatus: GameStatus;
    lastMoveAt: Date;
  };

  // Simple turn rotation for now
  // In a real game, this is complex (check win conditions, etc.)
  // But here we just rotate player and reset phase.

  const numPlayers = state.players.length;
  const nextPlayer = (state.currentPlayer % numPlayers) + 1; // 1-based player IDs

  // Skip eliminated players?
  // "Last Player Standing" rule.
  // For now, assume all players active.

  newState.currentPlayer = nextPlayer;
  newState.currentPhase = 'ring_placement'; // Default start of turn
  newState.lastMoveAt = new Date();

  return newState;
}

export function mutatePhaseChange(state: GameState, newPhase: GamePhase): GameState {
  const newState = {
    ...state,
    moveHistory: [...state.moveHistory],
  } as GameState & {
    currentPhase: GamePhase;
    lastMoveAt: Date;
  };

  newState.currentPhase = newPhase;
  newState.lastMoveAt = new Date();

  return newState;
}
