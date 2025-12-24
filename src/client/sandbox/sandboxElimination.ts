/**
 * @fileoverview Sandbox Elimination Helpers - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It provides elimination logic for sandbox/offline games.
 *
 * Canonical SSoT:
 * - Elimination logic: `src/shared/engine/aggregates/EliminationAggregate.ts`
 * - Elimination helpers: `src/shared/engine/aggregates/eliminationHelpers.ts`
 *
 * This adapter:
 * - Delegates to `eliminateFromStack()` from EliminationAggregate
 * - Uses `isStackEligibleForElimination()` for eligibility checks
 * - Provides `forceEliminateCapOnBoard()` for legacy sandbox flows
 * - Includes invariant assertions for debugging
 *
 * DO NOT add elimination rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type { BoardState, Player, RingStack, EliminationContext } from '../../shared/engine';
import {
  eliminateFromStack,
  isStackEligibleForElimination,
  calculateCapHeightElimination as calculateCapHeight,
} from '../../shared/engine';
import { flagEnabled, isTestEnvironment } from '../../shared/utils/envFlags';

const TERRITORY_TRACE_DEBUG = flagEnabled('RINGRIFT_TRACE_DEBUG');

export interface ForcedEliminationResult {
  board: BoardState;
  players: Player[];
  totalRingsEliminatedDelta: number;
}

function assertForcedEliminationConsistency(
  context: string,
  before: { board: BoardState; players: Player[] },
  after: { board: BoardState; players: Player[]; delta: number },
  playerNumber: number
): void {
  const isTestEnv = isTestEnvironment();

  const sumEliminated = (players: Player[]): number =>
    players.reduce((acc, p) => acc + p.eliminatedRings, 0);

  const sumBoardEliminated = (board: BoardState): number =>
    Object.values(board.eliminatedRings ?? {}).reduce((acc, v) => acc + v, 0);

  const beforePlayerTotal = sumEliminated(before.players);
  const beforeBoardTotal = sumBoardEliminated(before.board);
  const afterPlayerTotal = sumEliminated(after.players);
  const afterBoardTotal = sumBoardEliminated(after.board);

  const deltaPlayers = afterPlayerTotal - beforePlayerTotal;
  const deltaBoard = afterBoardTotal - beforeBoardTotal;

  const errors: string[] = [];

  if (deltaPlayers !== after.delta) {
    errors.push(
      `forced elimination (${context}) player delta mismatch: expected ${after.delta}, actual ${deltaPlayers}`
    );
  }

  if (deltaBoard !== after.delta) {
    errors.push(
      `forced elimination (${context}) board delta mismatch: expected ${after.delta}, actual ${deltaBoard}`
    );
  }

  if (after.delta < 0) {
    errors.push(
      `forced elimination (${context}) produced negative delta=${after.delta} for player ${playerNumber}`
    );
  }

  if (errors.length === 0) {
    return;
  }

  const message = `sandboxElimination invariant violation (${context}):` + '\n' + errors.join('\n');

  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
}

/**
 * Core elimination helper operating directly on the board and players.
 * This mirrors the logic in ClientSandboxEngine.forceEliminateCap but is
 * pure with respect to GameState, returning updated structures and the
 * number of rings eliminated.
 *
 * DELEGATES TO EliminationAggregate for canonical elimination semantics.
 *
 * Per RR-CANON-R022, R122, R145, R100:
 * - 'line': Eliminate exactly ONE ring from the top (any controlled stack is eligible)
 * - 'territory': Eliminate entire cap (any controlled stack eligible, including height-1)
 * - 'forced': Eliminate entire cap (any controlled stack is eligible)
 */
export function forceEliminateCapOnBoard(
  board: BoardState,
  players: Player[],
  playerNumber: number,
  stacks: RingStack[],
  eliminationContext: EliminationContext = 'forced'
): ForcedEliminationResult {
  const player = players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  if (stacks.length === 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  // Prefer stacks with positive stored capHeight (matches original behavior)
  // Then verify eligibility using canonical rules
  const stacksWithCap = stacks.filter((s) => s.capHeight > 0);
  const eligibleStack = stacksWithCap.find(
    (s) => isStackEligibleForElimination(s, eliminationContext, playerNumber).eligible
  );

  // Fall back to first stack with cap if no eligible found under current context
  const stack =
    eligibleStack ?? stacksWithCap[0] ?? stacks.find((s) => calculateCapHeight(s.rings) > 0);

  if (!stack || calculateCapHeight(stack.rings) <= 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  // If selected stack isn't eligible for requested context, use forced (least restrictive)
  const effectiveContext = isStackEligibleForElimination(stack, eliminationContext, playerNumber)
    .eligible
    ? eliminationContext
    : 'forced';

  return forceEliminateCapOnBoardInternal(board, players, playerNumber, stack, effectiveContext);
}

/**
 * Internal elimination function that delegates to EliminationAggregate.
 */
function forceEliminateCapOnBoardInternal(
  board: BoardState,
  players: Player[],
  playerNumber: number,
  stack: RingStack,
  eliminationContext: EliminationContext
): ForcedEliminationResult {
  if (TERRITORY_TRACE_DEBUG) {
    // eslint-disable-next-line no-console
    console.log('[sandboxElimination.forceEliminateCapOnBoard]', {
      playerNumber,
      stackPosition: stack.position,
      capHeight: calculateCapHeight(stack.rings),
      stackHeight: stack.stackHeight,
      eliminationContext,
    });
  }

  // Delegate to canonical EliminationAggregate
  const eliminationResult = eliminateFromStack({
    context: eliminationContext,
    player: playerNumber,
    stackPosition: stack.position,
    board,
  });

  if (!eliminationResult.success) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  // Update players array (EliminationAggregate only updates board)
  const updatedPlayers = players.map((p) =>
    p.playerNumber === playerNumber
      ? { ...p, eliminatedRings: p.eliminatedRings + eliminationResult.ringsEliminated }
      : p
  );

  const result: ForcedEliminationResult = {
    board: eliminationResult.updatedBoard,
    players: updatedPlayers,
    totalRingsEliminatedDelta: eliminationResult.ringsEliminated,
  };

  assertForcedEliminationConsistency(
    'forceEliminateCapOnBoard',
    { board, players },
    { board: result.board, players: result.players, delta: result.totalRingsEliminatedDelta },
    playerNumber
  );

  return result;
}
