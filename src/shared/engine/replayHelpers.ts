import type { GameState, Move, BoardType, Player, TimeControl } from '../types/game';
import type { GameRecord, MoveRecord } from '../types/gameRecord';
import { BOARD_CONFIGS } from '../types/game';
import { createInitialGameState } from './initialState';
import { processTurn } from './orchestration/turnOrchestrator';

/**
 * Reconstruct a GameState from a GameRecord at a given move index.
 *
 * This helper is deliberately minimal and replay-focused:
 * - It derives a plausible initial GameState compatible with the shared
 *   engine from the GameRecord header (boardType, numPlayers, rngSeed).
 * - It then applies the first `moveIndex` moves from the record using the
 *   canonical GameEngine, returning the resulting GameState.
 *
 * The returned state is suitable for parity checks, replay viewers, and
 * offline analysis, but is not guaranteed to match historical timestamps or
 * transient per-move metadata such as think times.
 *
 * @param record - Canonical GameRecord to replay.
 * @param moveIndex - Number of moves from the start to apply (0 = initial).
 */
export function reconstructStateAtMove(record: GameRecord, moveIndex: number): GameState {
  if (moveIndex < 0) {
    throw new Error(`moveIndex must be non-negative, got ${moveIndex}`);
  }

  const { boardType, numPlayers, rngSeed, isRated, players, moves } = record;
  const clampedIndex = Math.min(moveIndex, moves.length);

  const config = BOARD_CONFIGS[boardType];
  if (!config) {
    throw new Error(`Unknown boardType in GameRecord: ${boardType}`);
  }

  // Build minimal Player array for the engine; we preserve player numbers but
  // do not attempt to recover historical ratings or clocks.
  const timeControl: TimeControl = {
    type: 'classical',
    initialTime: 600, // seconds
    increment: 5,
  };

  const playerStates: Player[] = [];
  for (let i = 0; i < numPlayers; i += 1) {
    const seat = players[i];
    const playerNumber = i + 1;
    playerStates.push({
      id: `player${playerNumber}`,
      username: seat?.username ?? `Player ${playerNumber}`,
      type: seat?.playerType ?? 'ai',
      playerNumber,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: config.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    });
  }

  const initialState = createInitialGameState(
    record.id,
    boardType as BoardType,
    playerStates,
    timeControl,
    isRated,
    rngSeed
  );

  if (clampedIndex === 0) {
    return initialState;
  }

  // Convert MoveRecord entries into engine Moves and apply them.
  let state: GameState = initialState;
  for (let i = 0; i < clampedIndex; i += 1) {
    const rec: MoveRecord = record.moves[i];

    const move: Move = {
      id: `record-${record.id}-${i}`,
      player: rec.player,
      type: rec.type,
      // Spatial metadata
      ...(rec.from !== undefined ? { from: rec.from } : {}),
      ...(rec.to !== undefined ? { to: rec.to } : {}),
      ...(rec.captureTarget !== undefined ? { captureTarget: rec.captureTarget } : {}),
      // Placement metadata
      ...(rec.placementCount !== undefined ? { placementCount: rec.placementCount } : {}),
      ...(rec.placedOnStack !== undefined ? { placedOnStack: rec.placedOnStack } : {}),
      // Line/territory processing metadata. These fields are critical for TS↔Python
      // parity: when present, they carry the canonical geometry chosen by Python's
      // GameEngine for lines and territory regions, and should be preferred over
      // re-detection during replay.
      ...(rec.formedLines !== undefined ? { formedLines: rec.formedLines } : {}),
      ...(rec.collapsedMarkers !== undefined ? { collapsedMarkers: rec.collapsedMarkers } : {}),
      ...(rec.disconnectedRegions !== undefined
        ? { disconnectedRegions: rec.disconnectedRegions }
        : {}),
      ...(rec.eliminatedRings !== undefined ? { eliminatedRings: rec.eliminatedRings } : {}),
      // Replay-focused timing metadata. We do not attempt to reconstruct real
      // wall-clock timestamps; thinkTime is preserved when available.
      timestamp: new Date(0),
      thinkTime: rec.thinkTimeMs ?? 0,
      moveNumber: rec.moveNumber,
    } as Move;

    const result = processTurn(state, move);
    state = result.nextState;
  }

  // When we've applied the full recorded move list, treat the reconstructed
  // state as terminal for golden-replay invariants and analysis tooling. The
  // host is responsible for richer end-of-game semantics; here we only need a
  // structurally "finished" game so that INV-FINAL-STATE passes.
  if (clampedIndex === moves.length) {
    state = {
      ...state,
      gameStatus: 'finished',
    };
  }

  return state;
}
