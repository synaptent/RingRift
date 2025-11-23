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
  // The current selection policy depends only on phase and move types,
  // but we keep playerNumber in the signature for future per-player
  // tuning. Mark it as used to satisfy strict TS settings.
  void playerNumber;

  if (!candidates.length) {
    return null;
  }

  // Sort candidates deterministically to ensure RNG parity between engines
  // that might generate moves in different orders (e.g. RuleEngine vs Sandbox).
  candidates.sort((a, b) => {
    if (a.type !== b.type) return a.type.localeCompare(b.type);

    // Compare 'from'
    if (a.from && b.from) {
      if (a.from.x !== b.from.x) return a.from.x - b.from.x;
      if (a.from.y !== b.from.y) return a.from.y - b.from.y;
    } else if (a.from) return 1;
    else if (b.from) return -1;

    // Compare 'to'
    if (a.to && b.to) {
      if (a.to.x !== b.to.x) return a.to.x - b.to.x;
      if (a.to.y !== b.to.y) return a.to.y - b.to.y;
    } else if (a.to) return 1;
    else if (b.to) return -1;

    // Compare 'captureTarget'
    if (a.captureTarget && b.captureTarget) {
      if (a.captureTarget.x !== b.captureTarget.x) return a.captureTarget.x - b.captureTarget.x;
      if (a.captureTarget.y !== b.captureTarget.y) return a.captureTarget.y - b.captureTarget.y;
    } else if (a.captureTarget) return 1;
    else if (b.captureTarget) return -1;

    return 0;
  });

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
  // them explicitly. In earlier versions this was proportional to
  // counts; we now prioritise captures whenever they are available
  // so fallback play is tactically less blunder-prone.
  const captureMoves = nonPlacementMoves.filter(
    (m) => m.type === 'overtaking_capture' || m.type === 'continue_capture_segment'
  );

  const simpleMovementMoves = nonPlacementMoves.filter(
    (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
  );

  let pool: Move[] = [];

  // Movement / capture / chain_capture phases: always prefer captures
  // when at least one capture is available, otherwise fall back to
  // simple non-capturing moves. This keeps the policy deterministic
  // (up to RNG tie-breaking within a bucket) while strongly
  // prioritising immediate material gain.
  if (
    gameState.currentPhase === 'movement' ||
    gameState.currentPhase === 'capture' ||
    gameState.currentPhase === 'chain_capture'
  ) {
    if (captureMoves.length > 0) {
      pool = captureMoves;
    } else if (simpleMovementMoves.length > 0) {
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
