import { GameState, Move } from '../types/game';

/**
 * Shared local move-selection policy used by both the backend AI fallback
 * and the sandbox AI. Given a set of already-legal candidate moves for a
 * player, choose between placements, captures, and simple movements using
 * simple proportional weighting rather than hard preferences.
 *
 * Key behaviours:
 * - In movement / capture phases: randomly choose between capture moves
 *   and simple non-capturing moves in proportion to how many of each
 *   exist. Captures are *more likely* when plentiful, but not forced.
 * - In ring_placement: when both place_ring and non-placement moves
 *   (e.g. skip_placement) exist, randomly decide whether to place based
 *   on the ratio between the number of placement options and the number
 *   of non-placement moves.
 *
 * This helper is intentionally side-effect free so it can be used from
 * both server and client bundles without pulling in server-only logging
 * or services.
 */
export type LocalAIRng = () => number;

export function chooseLocalMoveFromCandidates(
  playerNumber: number,
  gameState: GameState,
  candidates: Move[],
  rng: LocalAIRng = Math.random
): Move | null {
  if (!candidates.length) {
    return null;
  }

  // Pre-bucket placement vs non-placement so we can reason about
  // placement decisions explicitly.
  const placementMoves = candidates.filter((m) => m.type === 'place_ring');
  const nonPlacementMoves = candidates.filter((m) => m.type !== 'place_ring');

  // === Ring placement weighting (place vs "do something else") ===
  //
  // When both placement and non-placement moves exist (e.g. optional
  // placement with skip_placement), decide probabilistically whether to
  // place a ring based on the ratio between the number of placement
  // options and the number of non-placement moves. Fewer placements vs
  // valid moves => lower chance of actually placing.
  if (
    gameState.currentPhase === 'ring_placement' &&
    placementMoves.length > 0 &&
    nonPlacementMoves.length > 0
  ) {
    const placementCount = placementMoves.length;
    const moveCount = nonPlacementMoves.length;
    const total = placementCount + moveCount;
    const r = rng() * total;

    const pool = r < placementCount ? placementMoves : nonPlacementMoves;
    const idx = Math.floor(rng() * pool.length);
    return pool[idx] ?? null;
  }

  // === Movement / capture weighting ===
  //
  // Separate movement/capture candidates so we can choose between
  // them in proportion to their counts, matching the sandbox AI
  // movement policy.
  const captureMoves = nonPlacementMoves.filter(
    (m) => m.type === 'overtaking_capture' || m.type === 'continue_capture_segment'
  );

  const simpleMovementMoves = nonPlacementMoves.filter(
    (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
  );

  let pool: Move[] = [];

  // Movement / capture / chain_capture phases: randomly choose between
  // captures and simple moves in proportion to how many of each exist.
  if (
    (gameState.currentPhase === 'movement' ||
      gameState.currentPhase === 'capture' ||
      gameState.currentPhase === 'chain_capture') &&
    (captureMoves.length > 0 || simpleMovementMoves.length > 0)
  ) {
    if (captureMoves.length > 0 && simpleMovementMoves.length > 0) {
      const total = captureMoves.length + simpleMovementMoves.length;
      const r = rng() * total;
      pool = r < captureMoves.length ? captureMoves : simpleMovementMoves;
    } else if (captureMoves.length > 0) {
      pool = captureMoves;
    } else {
      pool = simpleMovementMoves;
    }
  }

  // Ring placement phase fallback: if we have only placements (no
  // non-placement alternative), always place. This covers mandatory
  // placement cases where skip_placement is not legal.
  if (!pool.length && gameState.currentPhase === 'ring_placement' && placementMoves.length > 0) {
    pool = placementMoves;
  }

  // Global fallback: if we still have not selected a bucket (e.g. phases
  // with only skips or other special moves), use the full candidate set so
  // we always return a legal move.
  if (!pool.length) {
    pool = candidates;
  }

  const randomIndex = Math.floor(rng() * pool.length);
  return pool[randomIndex] ?? null;
}
