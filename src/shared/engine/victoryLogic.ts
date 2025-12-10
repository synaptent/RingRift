import { BoardState, BoardType, GameState, Position, positionToString } from '../types/game';
import { MovementBoardView, hasAnyLegalMoveOrCaptureFromOnBoard } from './core';
import { hasGlobalPlacementAction, hasForcedEliminationAction } from './globalActions';

export type VictoryReason =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'game_completed';

export interface VictoryResult {
  isGameOver: boolean;
  winner?: number;
  reason?: VictoryReason;
  /**
   * True when the result arises from a bare-board global stalemate where
   * rings remaining in hand are conceptually treated as eliminated for
   * tie-breaking purposes (FAQ Q11, §13.4).
   *
   * Hosts that persist elimination statistics (e.g. sandbox UI) may use
   * this flag to apply a hand→eliminated conversion before building a
   * final GameResult while keeping evaluateVictory itself side-effect free.
   */
  handCountsAsEliminated?: boolean;
}

/**
 * Canonical, side-effect-free victory evaluator for RingRift.
 *
 * Responsibilities:
 * - Primary victories:
 *   - Ring-elimination (Section 13.1) via GameState.victoryThreshold.
 *   - Territory-control (Section 13.2) via GameState.territoryVictoryThreshold.
 * - Structural terminality on a bare board:
 *   - If any player still has a legal ring placement (respecting the
 *     no-dead-placement rule via hasAnyLegalMoveOrCaptureFromOnBoard),
 *     the game is *not* over.
 *   - Otherwise apply the stalemate ladder (Section 13.4):
 *     territory → eliminated rings (including hand when appropriate) →
 *     markers → last actor → game_completed.
 *
 * This helper is shared by the backend RuleEngine, the shared GameEngine,
 * and the client sandbox so that victory semantics are defined in a single
 * place.
 */
export function evaluateVictory(state: GameState): VictoryResult {
  const players = state.players;

  if (!players || players.length === 0) {
    return { isGameOver: false };
  }

  // 1) Ring-elimination victory: strictly more than 50% of total rings in
  // play have been eliminated for a single player.
  const ringWinner = players.find((p) => p.eliminatedRings >= state.victoryThreshold);
  if (ringWinner) {
    return {
      isGameOver: true,
      winner: ringWinner.playerNumber,
      reason: 'ring_elimination',
      handCountsAsEliminated: false,
    };
  }

  // 2) Territory-control victory: strictly more than 50% of the board's
  // spaces are controlled as territory by a single player.
  const territoryWinner = players.find((p) => p.territorySpaces >= state.territoryVictoryThreshold);
  if (territoryWinner) {
    return {
      isGameOver: true,
      winner: territoryWinner.playerNumber,
      reason: 'territory_control',
      handCountsAsEliminated: false,
    };
  }

  // 3) Early Last-Player-Standing (R172): if exactly one player has stacks
  // on the board AND all other players have neither stacks nor rings in hand,
  // that player wins immediately.
  const playersWithStacks = new Set<number>();
  for (const stack of state.board.stacks.values()) {
    playersWithStacks.add(stack.controllingPlayer);
  }

  if (playersWithStacks.size === 1) {
    const stackOwner = Array.from(playersWithStacks)[0];
    const othersHaveMaterial = players.some(
      (p) =>
        p.playerNumber !== stackOwner &&
        (playersWithStacks.has(p.playerNumber) || p.ringsInHand > 0)
    );
    if (!othersHaveMaterial) {
      return {
        isGameOver: true,
        winner: stackOwner,
        reason: 'last_player_standing',
        handCountsAsEliminated: false,
      };
    }
  }

  // 4) Trapped position stalemate: All active players have stacks but no legal actions.
  // This handles AI vs AI games that stall when both players are blocked.
  // Per RR-CANON-R072/R100/R203: players with stacks can ALWAYS act - either with
  // real moves (placement/movement/capture) or via forced elimination.
  if (state.board.stacks.size > 0) {
    let somePlayerCanAct = false;

    for (const p of players) {
      let playerHasStacks = false;
      for (const stack of state.board.stacks.values()) {
        if (stack.controllingPlayer === p.playerNumber) {
          playerHasStacks = true;
          break;
        }
      }

      const hasMaterial = playerHasStacks || p.ringsInHand > 0;
      if (!hasMaterial) {
        continue;
      }

      if (hasGlobalPlacementAction(state, p.playerNumber)) {
        somePlayerCanAct = true;
        break;
      }

      // A player with stacks can always act: either with real moves (movement/capture)
      // or via forced elimination (RR-CANON-R072/R100). We only reach stalemate when
      // ALL players with stacks have neither real moves NOR forced elimination.
      if (playerHasStacks) {
        somePlayerCanAct = true;
        break;
      }
    }

    if (somePlayerCanAct) {
      return { isGameOver: false };
    }
  }

  // 5) Bare-board structural terminality & global stalemate.
  const anyRingsInHand = players.some((p) => p.ringsInHand > 0);
  let treatHandAsEliminated = false;
  let handCountsAsEliminated = false;

  if (anyRingsInHand) {
    // Check whether any player with rings in hand still has at least one
    // legal placement under the full no-dead-placement rule. If so, the
    // game is not yet structurally terminal: they may be able to re-enter
    // play once phases advance.
    const anyLegalPlacementForAnyPlayer = players.some(
      (p) => p.ringsInHand > 0 && hasAnyLegalPlacementOnBareBoard(state, p.playerNumber)
    );

    if (anyLegalPlacementForAnyPlayer) {
      return { isGameOver: false };
    }

    // Global stalemate on a bare board: some players hold rings in hand but
    // none of them has a legal placement. For tie-break purposes we
    // conceptually treat those rings as eliminated (hand → E) without
    // mutating the underlying GameState; hosts that care about persistent
    // elimination totals can mirror this via a separate helper.
    treatHandAsEliminated = true;
    handCountsAsEliminated = true;
  }

  // At this point the board has no stacks and either nobody holds rings in
  // hand or no legal placements exist for any player. Apply the stalemate
  // ladder.

  // First tie-breaker: territory spaces.
  const maxTerritory = Math.max(...players.map((p) => p.territorySpaces));
  const territoryLeaders = players.filter((p) => p.territorySpaces === maxTerritory);

  if (territoryLeaders.length === 1 && maxTerritory > 0) {
    return {
      isGameOver: true,
      winner: territoryLeaders[0].playerNumber,
      reason: 'territory_control',
      handCountsAsEliminated,
    };
  }

  // Second tie-breaker: eliminated rings, including rings remaining in hand
  // when we are in a global-stalemate case.
  const eliminationScores = players.map(
    (p) => p.eliminatedRings + (treatHandAsEliminated ? p.ringsInHand : 0)
  );
  const maxEliminated = Math.max(...eliminationScores);
  const eliminationLeaders = players.filter((_, idx) => {
    return eliminationScores[idx] === maxEliminated;
  });

  if (eliminationLeaders.length === 1 && maxEliminated > 0) {
    return {
      isGameOver: true,
      winner: eliminationLeaders[0].playerNumber,
      reason: 'ring_elimination',
      handCountsAsEliminated,
    };
  }

  // Third tie-breaker: remaining markers on the board.
  const markerCountsByPlayer: { [player: number]: number } = {};
  for (const p of players) {
    markerCountsByPlayer[p.playerNumber] = 0;
  }
  for (const marker of state.board.markers.values()) {
    const owner = marker.player;
    if (markerCountsByPlayer[owner] !== undefined) {
      markerCountsByPlayer[owner] += 1;
    }
  }

  const markerCounts = players.map((p) => markerCountsByPlayer[p.playerNumber] ?? 0);
  const maxMarkers = Math.max(...markerCounts);
  const markerLeaders = players.filter(
    (p) => (markerCountsByPlayer[p.playerNumber] ?? 0) === maxMarkers
  );

  if (markerLeaders.length === 1 && maxMarkers > 0) {
    return {
      isGameOver: true,
      winner: markerLeaders[0].playerNumber,
      reason: 'last_player_standing',
      handCountsAsEliminated,
    };
  }

  // Final tie-breaker: last player to complete a valid turn action.
  const lastActor = getLastActor(state);
  if (lastActor !== undefined) {
    return {
      isGameOver: true,
      winner: lastActor,
      reason: 'last_player_standing',
      handCountsAsEliminated,
    };
  }

  // Safety fallback: in degenerate cases where no last actor can be
  // determined (e.g. malformed game state), mark the game as completed
  // without a specific winner.
  return {
    isGameOver: true,
    reason: 'game_completed',
    handCountsAsEliminated,
  };
}

/**
 * Determine the last player to complete a valid turn action, used as the
 * final rung of the stalemate tie-break ladder. Preference order:
 *
 * 1. The actor of the last structured history entry, when available.
 * 2. The player of the last legacy moveHistory entry.
 * 3. The player immediately preceding currentPlayer in turn order.
 */
export function getLastActor(state: GameState): number | undefined {
  if (state.history && state.history.length > 0) {
    const lastEntry = state.history[state.history.length - 1];
    if (lastEntry && typeof lastEntry.actor === 'number') {
      return lastEntry.actor;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastMove = state.moveHistory[state.moveHistory.length - 1];
    if (lastMove && typeof lastMove.player === 'number') {
      return lastMove.player;
    }
  }

  const players = state.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  const currentIdx = players.findIndex((p) => p.playerNumber === state.currentPlayer);
  if (currentIdx === -1) {
    return players[0].playerNumber;
  }

  const lastIdx = (currentIdx - 1 + players.length) % players.length;
  return players[lastIdx].playerNumber;
}

/**
 * Check whether the specified player has at least one legal ring placement
 * on a bare board (no stacks present) that satisfies the no-dead-placement
 * rule. This is used only for structural terminality detection; it is not
 * a general-purpose placement enumerator.
 */
function hasAnyLegalPlacementOnBareBoard(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player || player.ringsInHand <= 0) {
    return false;
  }

  const board = state.board;
  const boardType: BoardType = board.type;

  // Precondition: callers only invoke this when there are no stacks on the
  // board. Defensively handle unexpected stacks by delegating to the shared
  // movement helper anyway.
  const hasStacks = board.stacks.size > 0;

  let found = false;
  forEachBoardPosition(board, (pos) => {
    if (found) {
      return;
    }

    const key = positionToString(pos);

    // Do not allow placement on collapsed territory or markers; stacks and
    // markers are mutually exclusive with territory under the board
    // invariants.
    if (board.collapsedSpaces.has(key)) {
      return;
    }

    if (board.markers.has(key)) {
      return;
    }

    // Build a minimal MovementBoardView over a hypothetical board that has
    // a single new stack at `pos` for the given player. When the board
    // unexpectedly contains stacks, we also expose them to the view so
    // no-dead-placement remains conservative.
    const hypotheticalKey = key;

    const view: MovementBoardView = {
      isValidPosition: (p) => isValidBoardPosition(boardType, board.size, p),
      isCollapsedSpace: (p) => {
        const k = positionToString(p);
        return board.collapsedSpaces.has(k);
      },
      getStackAt: (p) => {
        const k = positionToString(p);
        if (k === hypotheticalKey) {
          return {
            controllingPlayer: playerNumber,
            capHeight: 1,
            stackHeight: 1,
          };
        }
        if (!hasStacks) {
          return undefined;
        }
        const existing = board.stacks.get(k);
        if (!existing) {
          return undefined;
        }
        return {
          controllingPlayer: existing.controllingPlayer,
          capHeight: existing.capHeight,
          stackHeight: existing.stackHeight,
        };
      },
      getMarkerOwner: (p) => {
        const k = positionToString(p);
        const marker = board.markers.get(k);
        return marker?.player;
      },
    };

    if (hasAnyLegalMoveOrCaptureFromOnBoard(boardType, pos, playerNumber, view)) {
      found = true;
    }
  });

  return found;
}

function forEachBoardPosition(board: BoardState, fn: (pos: Position) => void): void {
  const size = board.size;

  if (board.type === 'hexagonal') {
    const radius = size - 1;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        fn({ x: q, y: r, z: s });
      }
    }
  } else {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        fn({ x, y });
      }
    }
  }
}

function isValidBoardPosition(boardType: BoardType, size: number, position: Position): boolean {
  if (boardType === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z ?? -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  }

  return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
}
