import sqlite3 from 'better-sqlite3';
import { enumerateProcessLineMoves } from '../src/shared/engine/aggregates/LineAggregate';
import { hasPhaseLocalInteractiveMove, isANMState, hasTurnMaterial, hasGlobalPlacementAction, hasForcedEliminationAction } from '../src/shared/engine/globalActions';
import type { GameState } from '../src/shared/types/game';

const db = sqlite3('ai-service/data/games/canonical_square8.db');
const gameId = '907239ab-89db-4014-aa35-cac4e3aa9255';

// Get state JSON at move index 113
const row = db.prepare(`
  SELECT state_json FROM state_history 
  WHERE game_id = ? AND move_index = ?
`).get(gameId, 113) as { state_json: string } | undefined;

if (!row) {
  console.log('State not found');
  process.exit(1);
}

const state: GameState = JSON.parse(row.state_json);

console.log('currentPlayer:', state.currentPlayer);
console.log('currentPhase:', state.currentPhase);
console.log('gameStatus:', state.gameStatus);

const moves = enumerateProcessLineMoves(state, state.currentPlayer, { detectionMode: 'detect_now' });
console.log('enumerateProcessLineMoves returns', moves.length, 'moves');
if (moves.length > 0) {
  console.log('moves:', moves.map(m => m.type));
}

console.log('hasTurnMaterial:', hasTurnMaterial(state, state.currentPlayer));
console.log('hasGlobalPlacementAction:', hasGlobalPlacementAction(state, state.currentPlayer));
console.log('hasPhaseLocalInteractiveMove:', hasPhaseLocalInteractiveMove(state, state.currentPlayer));
console.log('hasForcedEliminationAction:', hasForcedEliminationAction(state, state.currentPlayer));
console.log('isANMState:', isANMState(state));

db.close();
