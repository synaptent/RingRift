import type { GameState, Move, Position, LineInfo, BoardType } from '../types/game';
import { positionToString } from '../types/game';
import { findAllLines } from './lineDetection';
import { getEffectiveLineLengthThreshold } from './rulesConfig';

/**
 * Shared helpers for line-processing decision enumeration and application.
 *
 * This module is the intended single source of truth for:
 *
 * - Which `process_line` and `choose_line_reward` moves are available in a
 *   given `GameState` for a player, and
 * - How those moves should update the board and per-player bookkeeping when
 *   applied.
 *
 * It is designed to absorb the remaining duplication between:
 *
 * - Backend:
 *   - `GameEngine.getValidLineProcessingMoves` and
 *   - [`processLinesForCurrentPlayer`](src/server/game/rules/lineProcessing.ts:1),
 * - Sandbox:
 *   - `sandboxLinesEngine.getValidLineProcessingMoves`, and
 *   - `sandboxLinesEngine.applyLineDecisionMove` /
 *     `ClientSandboxEngine.processLinesForCurrentPlayer`.
 *
 * Geometry (which cells form a line) is delegated to the shared helpers in
 * {@link lineDetection.ts}. The helpers in this module focus on **decision
 * surfaces and state updates**: collapsing markers to territory, returning
 * rings from stacks on those spaces to hand, and flagging when a mandatory
 * self-elimination reward must follow.
 */

/**
 * Options that control how line-processing moves are enumerated.
 */
export interface LineEnumerationOptions {
  /**
   * Whether to re-run line detection from scratch over the current board
   * state (`detect_now`) or to trust `state.board.formedLines`
   * (`use_board_cache`).
   *
   * - In the backend GameEngine, lines are typically detected immediately
   *   after a movement/capture step and cached on the board.
   * - In some sandbox flows, line detection may be re-run on demand.
   *
   * Default: 'use_board_cache'.
   */
  detectionMode?: 'use_board_cache' | 'detect_now';

  /**
   * When `detectionMode === 'detect_now'`, controls which board type /
   * configuration should be used for line-adjacency rules. In almost all
   * cases this is simply `state.board.type`.
   */
  boardTypeOverride?: BoardType;
}

/**
 * Compute the next canonical moveNumber for decision moves based on the
 * existing history/moveHistory. This keeps numbering stable across hosts
 * without requiring callers to thread an explicit counter.
 */
function computeNextMoveNumber(state: GameState): number {
  if (state.history && state.history.length > 0) {
    const last = state.history[state.history.length - 1];
    if (typeof last.moveNumber === 'number' && last.moveNumber > 0) {
      return last.moveNumber + 1;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastLegacy = state.moveHistory[state.moveHistory.length - 1];
    if (typeof lastLegacy.moveNumber === 'number' && lastLegacy.moveNumber > 0) {
      return lastLegacy.moveNumber + 1;
    }
  }

  return 1;
}

function getEffectiveBoardType(state: GameState, options?: LineEnumerationOptions): BoardType {
  return (options?.boardTypeOverride ?? state.board.type) as BoardType;
}

/**
 * Detect all lines for the given player according to the chosen enumeration
 * options. Geometry is delegated to {@link findAllLines}; when
 * `detectionMode === 'use_board_cache'` and `board.formedLines` is non-empty,
 * that cache is preferred instead.
 */
function detectPlayerLines(
  state: GameState,
  player: number,
  options?: LineEnumerationOptions
): LineInfo[] {
  const board = state.board;
  const mode = options?.detectionMode ?? 'use_board_cache';

  if (mode === 'use_board_cache' && board.formedLines && board.formedLines.length > 0) {
    return board.formedLines.filter((line) => line.player === player);
  }

  // Fresh detection; filter by owner.
  return findAllLines(board).filter((line) => line.player === player);
}

/**
 * Canonical, order-independent key for a line based on its marker positions.
 * This is used to match a Move's embedded `formedLines[0]` back to the
 * current board geometry even if detection order changes.
 */
function canonicalLineKey(line: LineInfo): string {
  return line.positions
    .map((p) => positionToString(p))
    .sort()
    .join('|');
}

/**
 * Resolve the concrete LineInfo on the current board that a decision Move is
 * referring to. Preference order:
 *
 * 1. A line whose canonical position-set matches `move.formedLines[0]`.
 * 2. Fallback to the first line for `move.player` when no exact match exists.
 */
function resolveLineForMove(
  state: GameState,
  move: Move,
  options?: LineEnumerationOptions
): LineInfo | undefined {
  const player = move.player;
  const playerLines = detectPlayerLines(state, player, options);
  if (playerLines.length === 0) {
    return undefined;
  }

  if (move.formedLines && move.formedLines.length > 0) {
    const target = move.formedLines[0];
    const targetKey = canonicalLineKey(target);

    const matched = playerLines.find((line) => canonicalLineKey(line) === targetKey);
    if (matched) {
      return matched;
    }
  }

  // Fallback: preserve historical behaviour of "first line wins" when no
  // precise match can be found.
  return playerLines[0];
}

/**
 * Collapse the given marker positions into territory for `player`, returning
 * rings from any stacks on those spaces to the appropriate players' hands.
 *
 * This helper:
 * - clones the board maps and players array (shallowly),
 * - removes stacks/markers on the collapsed spaces,
 * - marks those spaces as collapsed territory for `player`, and
 * - increments `territorySpaces` for `player` by the number of unique
 *   positions collapsed.
 *
 * It intentionally does **not** change `totalRingsInPlay` or any eliminated
 * counters; rings returned from stacks remain "in play" and are simply moved
 * back to hand.
 */
function collapseLinePositions(
  state: GameState,
  positions: Position[],
  player: number
): { nextState: GameState; collapsedCount: number } {
  const board = state.board;

  const nextBoard = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };

  const nextPlayers = state.players.map((p) => ({ ...p }));
  const collapsedKeys = new Set<string>();

  for (const pos of positions) {
    const key = positionToString(pos);
    if (collapsedKeys.has(key)) {
      continue;
    }
    collapsedKeys.add(key);

    // Return any rings on this space to their owners' hands, then remove the
    // stack entirely. This matches the "returned to supply" semantics from
    // the line rules (as opposed to elimination).
    const stack = nextBoard.stacks.get(key);
    if (stack && Array.isArray(stack.rings) && stack.rings.length > 0) {
      for (const ringOwner of stack.rings as number[]) {
        const idx = nextPlayers.findIndex((p) => p.playerNumber === ringOwner);
        if (idx >= 0) {
          const current = nextPlayers[idx];
          nextPlayers[idx] = {
            ...current,
            ringsInHand: (current.ringsInHand ?? 0) + 1,
          };
        }
      }
      nextBoard.stacks.delete(key);
    }

    // Remove any marker at this position.
    if (nextBoard.markers.has(key)) {
      nextBoard.markers.delete(key);
    }

    // Mark as collapsed territory for the acting player.
    nextBoard.collapsedSpaces.set(key, player);
  }

  if (collapsedKeys.size > 0) {
    const idx = nextPlayers.findIndex((p) => p.playerNumber === player);
    if (idx >= 0) {
      const current = nextPlayers[idx];
      nextPlayers[idx] = {
        ...current,
        territorySpaces: current.territorySpaces + collapsedKeys.size,
      };
    }
  }

  // Drop any cached formedLines that intersect collapsed spaces; callers that
  // care about further lines should re-run detection via findAllLines.
  if (nextBoard.formedLines && nextBoard.formedLines.length > 0) {
    nextBoard.formedLines = nextBoard.formedLines.filter((line) => {
      return !line.positions.some((p) => collapsedKeys.has(positionToString(p)));
    });
  }

  const nextState: GameState = {
    ...state,
    board: nextBoard,
    players: nextPlayers,
  };

  return { nextState, collapsedCount: collapsedKeys.size };
}

/**
 * Enumerate `process_line` decision moves for the specified player in the
 * current `GameState`.
 *
 * Semantics:
 *
 * - Only lines owned by `player` are considered.
 * - Line geometry is derived either from `state.board.formedLines` or by
 *   invoking the shared `findAllLines` helper in
 *   [`lineDetection.ts`](src/shared/engine/lineDetection.ts:1), depending on
 *   {@link LineEnumerationOptions.detectionMode}.
 * - Each returned {@link Move} has:
 *   - `type: 'process_line'`,
 *   - `player: player`,
 *   - `formedLines[0]` containing the selected line, and
 *   - `to` set to a representative position on that line (the first position).
 *
 * Error handling:
 *
 * - This helper is pure and does not mutate the input state.
 * - It assumes that board invariants hold (no stack/marker/territory overlap).
 * - For malformed states it is permitted to throw, but must be total for
 *   states produced by the canonical engines.
 */
export function enumerateProcessLineMoves(
  state: GameState,
  player: number,
  options?: LineEnumerationOptions
): Move[] {
  const playerLines = detectPlayerLines(state, player, options);

  if (playerLines.length === 0) {
    return [];
  }

  const boardType = getEffectiveBoardType(state, options);
  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );

  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  playerLines.forEach((line, index) => {
    // Filter out lines that do not meet the effective threshold (e.g. 3-in-a-row
    // on 2p 8x8).
    if (line.length < requiredLength) {
      return;
    }

    const representative = line.positions[0] ?? { x: 0, y: 0 };
    const lineKey = line.positions.map((p) => positionToString(p)).join('|');

    moves.push({
      id: `process-line-${index}-${lineKey}`,
      type: 'process_line',
      player,
      to: representative,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  });

  return moves;
}
/**
 * Enumerate `choose_line_reward` decision moves for a specific line that has
 * already been selected for processing.
 *
 * Typical usage pattern:
 *
 * 1. Call {@link enumerateProcessLineMoves} and allow the player/AI to choose
 *    a `process_line` move (possibly via a PlayerChoice of type `line_order`).
 * 2. Apply the chosen `process_line` move using
 *    {@link applyProcessLineDecision} to update the board and advance the
 *    line-processing state.
 * 3. If the processed line is longer than the minimum threshold, call
 *    `enumerateChooseLineRewardMoves` to present:
 *    - Option 1: collapse the entire line and earn a ring-elimination reward;
 *    - Option 2: collapse a minimum contiguous subset of markers (no reward).
 *
 * Semantics:
 *
 * - For a line whose length equals the board-configured minimum (`L`):
 *   - Only an implicit "collapse all" option exists; engines may choose to
 *     represent this either as a `process_line` move with no subsequent
 *     `choose_line_reward`, or as a single `choose_line_reward` move with
 *     `selection: 'COLLAPSE_ALL'`.
 * - For a line longer than `L`:
 *   - One `choose_line_reward` move with collapse-all semantics.
 *   - One or more `choose_line_reward` moves with minimum-collapse semantics
 *     and `collapsedMarkers` set to each legal contiguous segment of length
 *     `L` along the line.
 *
 * This helper does **not** apply any rewards or board changes; callers must
 * pass the chosen move to {@link applyChooseLineRewardDecision}.
 */
export function enumerateChooseLineRewardMoves(
  state: GameState,
  player: number,
  lineIndex: number
): Move[] {
  if (lineIndex < 0) {
    return [];
  }

  const playerLines = detectPlayerLines(state, player);
  if (playerLines.length === 0 || lineIndex >= playerLines.length) {
    return [];
  }

  const line = playerLines[lineIndex];
  const boardType = getEffectiveBoardType(state);
  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );

  if (!line.positions || line.positions.length === 0) {
    return [];
  }

  if (line.length < requiredLength) {
    // Not a complete, collapsible line – no reward moves.
    return [];
  }

  const representative = line.positions[0] ?? { x: 0, y: 0 };
  const lineKey = line.positions.map((p) => positionToString(p)).join('|');
  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  // Exact-length line: a single collapse-all variant for hosts that prefer to
  // express this as a dedicated reward choice.
  if (line.length === requiredLength) {
    moves.push({
      id: `choose-line-reward-${lineIndex}-${lineKey}-all`,
      type: 'choose_line_reward',
      player,
      to: representative,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);

    return moves;
  }

  // Overlength line (> requiredLength).
  // Option 1: collapse the entire line.
  moves.push({
    id: `choose-line-reward-${lineIndex}-${lineKey}-all`,
    type: 'choose_line_reward',
    player,
    to: representative,
    formedLines: [line],
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: nextMoveNumber,
  } as Move);

  // Option 2: all legal minimum-collapse segments of length L along the line.
  const maxStart = line.length - requiredLength;
  for (let start = 0; start <= maxStart; start++) {
    const segment = line.positions.slice(start, start + requiredLength);

    moves.push({
      id: `choose-line-reward-${lineIndex}-${lineKey}-min-${start}`,
      type: 'choose_line_reward',
      player,
      to: representative,
      formedLines: [line],
      collapsedMarkers: segment,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  }

  return moves;
}

/**
 * Result of applying a line-processing decision.
 */
export interface LineDecisionApplicationOutcome {
  /**
   * Next GameState after applying the chosen decision, including:
   *
   * - collapse of markers to territory where appropriate;
   * - updates to `board.collapsedSpaces`, `board.markers`, and
   *   `players[n].territorySpaces`;
   * - any rings returned to hand when stacks are removed from collapsed
   *   spaces; and
   * - updates to `board.formedLines` when processed or broken lines are
   *   removed.
   */
  nextState: GameState;

  /**
   * When true, this decision granted the acting player a mandatory
   * self-elimination reward that must be paid via a follow-up
   * `eliminate_rings_from_stack` move. Under current backend/sandbox
   * semantics this is the case for:
   *
   * - exact-length lines processed via `process_line`, and
   * - overlength lines when using the collapse-all reward option.
   *
   * The choice of *where* to eliminate from and how that elimination is
   * surfaced (explicit move vs automatic from hand) remains a host concern;
   * this flag exists purely to keep bookkeeping consistent across engines.
   */
  pendingLineRewardElimination: boolean;
}

/**
 * Apply a `process_line` move produced by
 * {@link enumerateProcessLineMoves} to the given `GameState`.
 *
 * Responsibilities:
 *
 * - Collapse the selected line's markers according to the implicit exact-
 *   length behaviour: collapse the entire line to territory and credit a
 *   mandatory self-elimination reward to the acting player.
 * - Remove any stacks and markers on the collapsed spaces and update per-
 *   player ring-in-hand counts when rings are returned from stacks.
 * - Update `board.collapsedSpaces` and `players[n].territorySpaces` for the
 *   acting player.
 * - Remove the processed line from `board.formedLines` and discard any other
 *   lines that are broken by the collapse, mirroring backend and sandbox
 *   semantics.
 *
 * For overlength lines, callers should prefer
 * {@link applyChooseLineRewardDecision}; this helper treats `process_line`
 * on an overlength line as a no-op to avoid surprising behaviour.
 */
export function applyProcessLineDecision(
  state: GameState,
  move: Move
): LineDecisionApplicationOutcome {
  if (move.type !== 'process_line') {
    throw new Error(
      `applyProcessLineDecision expected move.type === 'process_line', got '${move.type}'`
    );
  }

  const line = resolveLineForMove(state, move);
  if (!line) {
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  const boardType = state.board.type as BoardType;
  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );

  if (line.length < requiredLength) {
    // Not actually a complete line; treat defensively as a no-op.
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  if (line.length > requiredLength) {
    // Defensive: overlength reward semantics are expressed via
    // choose_line_reward. To avoid surprising marker changes when callers
    // accidentally pass such a line here, treat as a no-op.
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  const { nextState } = collapseLinePositions(state, line.positions, move.player);

  return {
    nextState,
    // Exact-length line processing always grants a mandatory self-elimination
    // reward under current semantics, modelled as a follow-up
    // eliminate_rings_from_stack decision.
    pendingLineRewardElimination: true,
  };
}

/**
 * Apply a `choose_line_reward` move produced by
 * {@link enumerateChooseLineRewardMoves} to the given `GameState`.
 *
 * Responsibilities:
 *
 * - For collapse-all variants:
 *   - collapse the entire line to territory for the acting player;
 *   - return any rings on those spaces to their owners' hands; and
 *   - set `pendingLineRewardElimination: true` when the line is at least the
 *     minimum length required for collapse.
 * - For minimum-collapse variants:
 *   - collapse only the specified contiguous `collapsedMarkers` subset;
 *   - update territory and ring-in-hand counts accordingly; and
 *   - ensure `pendingLineRewardElimination: false` (no additional reward).
 *
 * As with {@link applyProcessLineDecision}, this helper is pure with respect
 * to its inputs and returns a new GameState instance.
 */
export function applyChooseLineRewardDecision(
  state: GameState,
  move: Move
): LineDecisionApplicationOutcome {
  if (move.type !== 'choose_line_reward') {
    throw new Error(
      `applyChooseLineRewardDecision expected move.type === 'choose_line_reward', got '${move.type}'`
    );
  }

  const line = resolveLineForMove(state, move);
  if (!line) {
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  const boardType = state.board.type as BoardType;
  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );
  const length = line.length;

  if (length < requiredLength) {
    // Not a complete, collapsible line; treat as no-op.
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  let positionsToCollapse: Position[] = line.positions;
  let pendingReward = false;

  if (length === requiredLength) {
    // Exact-length line: collapse all markers and treat this as granting a
    // mandatory self-elimination reward.
    positionsToCollapse = line.positions;
    pendingReward = true;
  } else {
    // Overlength line.
    const collapsed = move.collapsedMarkers;
    const isCollapseAll =
      !collapsed ||
      collapsed.length === length ||
      // Defensive: any subset that claims to collapse ≥ length markers is
      // treated as collapse-all.
      collapsed.length > length;

    if (isCollapseAll) {
      positionsToCollapse = line.positions;
      pendingReward = true;
    } else if (collapsed) {
      // Minimum-collapse option: collapse exactly the selected subset.
      positionsToCollapse = collapsed;
      pendingReward = false;
    }
  }

  const { nextState } = collapseLinePositions(state, positionsToCollapse, move.player);

  return {
    nextState,
    pendingLineRewardElimination: pendingReward,
  };
}
