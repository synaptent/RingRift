import type { BoardState, GameState } from '../../shared/types/game';

function mapToRecord<V>(map: Map<string, V>): Record<string, V> {
  const out: Record<string, V> = {};
  for (const [key, value] of map.entries()) {
    out[key] = value;
  }
  return out;
}

function toPythonWireBoardState(board: BoardState): Record<string, unknown> {
  return {
    ...board,
    stacks: mapToRecord(board.stacks),
    markers: mapToRecord(board.markers),
    collapsedSpaces: mapToRecord(board.collapsedSpaces),
    territories: mapToRecord(board.territories),
  };
}

/**
 * Convert a Map-backed TS GameState into a JSON-wire-safe structure for the
 * Python AI service. Axios uses JSON.stringify under the hood, which drops
 * Map contents; we must convert Maps to plain objects before sending.
 */
export function toPythonWireGameState(gameState: GameState): Record<string, unknown> {
  return {
    ...gameState,
    board: toPythonWireBoardState(gameState.board),
  };
}

export function toPythonWireGameStateOptional(
  gameState: GameState | null | undefined
): Record<string, unknown> | undefined {
  if (!gameState) {
    return undefined;
  }
  return toPythonWireGameState(gameState);
}
