import { GameState, Move } from '../../src/shared/types/game';
import { GameEngine } from '../../src/server/game/GameEngine';
import { CanonicalReplayEngine } from '../../src/shared/replay';
import { createBackendEngineFromInitialState } from './traces';
import { findMatchingBackendMove } from './moveMatching';
import { snapshotFromGameState, snapshotsEqual, ComparableSnapshot } from './stateSnapshots';

/**
 * Minimal shared trace replay helpers so parity harnesses can agree on a
 * single canonical "apply move → post-move lifecycle → snapshot" model
 * for both backend and sandbox engines.
 */

export interface EngineHandle {
  /** Return a defensive snapshot of the current GameState. */
  getState(): GameState;

  /**
   * Apply a canonical Move. Implementations are responsible for invoking
   * any post-move lifecycle (e.g. advanceAfterMovement/stepAutomaticPhases)
   * so that the resulting state matches the semantics used during trace
   * generation.
   *
   * Returns true if the engine state changed as a result of the move.
   */
  applyCanonicalMove(move: Move): Promise<boolean> | boolean;
}

export interface EngineAdapter {
  /** Human-readable name for logging. */
  name: string;

  /** Construct a fresh engine from an initial GameState. */
  create(initial: GameState): Promise<EngineHandle> | EngineHandle;
}

export interface ReplayResult {
  /** Whether final snapshots for the full move list were equal. */
  allEqual: boolean;
  /** Index of first mismatching prefix in [0, moves.length], or moves.length when none. */
  firstMismatchIndex: number;
}

class BackendEngineHandle implements EngineHandle {
  constructor(private engine: GameEngine) {}

  getState(): GameState {
    return this.engine.getGameState();
  }

  async applyCanonicalMove(move: Move): Promise<boolean> {
    const before = this.engine.getGameState();
    const valid = this.engine.getValidMoves(before.currentPlayer);
    const matching = findMatchingBackendMove(move, valid);

    if (!matching) {
      // No matching backend move for this sandbox action at this point in
      // the replay. Callers can treat this as a divergence.
      // We deliberately log minimally here; higher-level harnesses may emit
      // richer diagnostics once the mismatching prefix has been found.

      console.error('[TraceReplayer.Backend] No matching backend move', {
        sandboxMove: {
          type: move.type,
          player: move.player,
          from: move.from,
          to: move.to,
          captureTarget: move.captureTarget,
          moveNumber: move.moveNumber,
        },
        backendCurrentPlayer: before.currentPlayer,
        backendCurrentPhase: before.currentPhase,
        backendValidMovesCount: valid.length,
      });
      return false;
    }

    const { id, timestamp, moveNumber, ...payload } = matching as any;
    const result = await this.engine.makeMove(
      payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
    );

    if (!result.success) {
      console.error('[TraceReplayer.Backend] makeMove failed', {
        backendMoveNumber: (matching as any).moveNumber,
        error: result.error,
      });
      return false;
    }

    // Mirror the existing bisect/internalStateParity harness semantics by
    // stepping through any remaining automatic bookkeeping phases so the
    // backend reaches the same post-move boundary as the sandbox.
    await this.engine.stepAutomaticPhasesForTesting();

    const after = this.engine.getGameState();
    return JSON.stringify(before) !== JSON.stringify(after);
  }
}

class ReplayEngineHandle implements EngineHandle {
  constructor(private engine: CanonicalReplayEngine) {}

  getState(): GameState {
    return this.engine.getState() as GameState;
  }

  async applyCanonicalMove(move: Move): Promise<boolean> {
    const before = this.engine.getState();
    const result = await this.engine.applyMove(move);
    const after = this.engine.getState();
    return result.success && JSON.stringify(before) !== JSON.stringify(after);
  }
}

export const backendAdapter: EngineAdapter = {
  name: 'backend',
  create(initial: GameState): EngineHandle {
    const engine = createBackendEngineFromInitialState(initial);
    return new BackendEngineHandle(engine);
  },
};

export const sandboxAdapter: EngineAdapter = {
  name: 'canonical-replay',
  create(initial: GameState): EngineHandle {
    const engine = new CanonicalReplayEngine({
      gameId: initial.id,
      boardType: initial.boardType,
      numPlayers: initial.players.length,
      initialState: initial,
    });
    return new ReplayEngineHandle(engine);
  },
};

/**
 * Replay the first {@code prefixLength} moves on a fresh engine constructed
 * from {@code initial}, returning the handle and its final state.
 */
export async function replayPrefixOnEngine(
  adapter: EngineAdapter,
  initial: GameState,
  moves: Move[],
  prefixLength: number
): Promise<{ engine: EngineHandle; finalState: GameState }> {
  const engine = await adapter.create(initial);

  for (let i = 0; i < prefixLength; i++) {
    const move = moves[i];
    await engine.applyCanonicalMove(move);
  }

  return { engine, finalState: engine.getState() };
}

/**
 * Replay a prefix on backend and sandbox engines and compare their
 * snapshots via the shared ComparableSnapshot helpers.
 */
export async function compareEnginesAtPrefix(
  backend: EngineAdapter,
  sandbox: EngineAdapter,
  initial: GameState,
  moves: Move[],
  prefixLength: number
): Promise<{
  equal: boolean;
  backendSnap: ComparableSnapshot;
  sandboxSnap: ComparableSnapshot;
}> {
  if (prefixLength === 0) {
    const snap = snapshotFromGameState('prefix-0', initial);
    return { equal: true, backendSnap: snap, sandboxSnap: snap };
  }

  const { finalState: backendState } = await replayPrefixOnEngine(
    backend,
    initial,
    moves,
    prefixLength
  );
  const { finalState: sandboxState } = await replayPrefixOnEngine(
    sandbox,
    initial,
    moves,
    prefixLength
  );

  const backendSnap = snapshotFromGameState(`backend-prefix-${prefixLength}`, backendState);
  const sandboxSnap = snapshotFromGameState(`sandbox-prefix-${prefixLength}`, sandboxState);

  return {
    equal: snapshotsEqual(backendSnap, sandboxSnap),
    backendSnap,
    sandboxSnap,
  };
}

/**
 * Binary search for the smallest prefix length at which backend and
 * sandbox snapshots diverge when replaying {@code moves} from
 * {@code initial}.
 */
export async function findFirstMismatchIndex(
  backend: EngineAdapter,
  sandbox: EngineAdapter,
  initial: GameState,
  moves: Move[]
): Promise<ReplayResult> {
  const statesEqualAt = async (k: number): Promise<boolean> => {
    const { equal } = await compareEnginesAtPrefix(backend, sandbox, initial, moves, k);
    return equal;
  };

  // Sanity: prefix length 0 must be equal by construction.
  if (!(await statesEqualAt(0))) {
    return { allEqual: false, firstMismatchIndex: 0 };
  }

  let lo = 0;
  let hi = moves.length;

  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const equal = await statesEqualAt(mid);
    if (equal) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const firstMismatchIndex = lo;
  const allEqual = await statesEqualAt(moves.length);

  return { allEqual, firstMismatchIndex };
}
