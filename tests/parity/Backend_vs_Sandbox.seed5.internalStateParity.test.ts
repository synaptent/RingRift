import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { snapshotFromGameState, snapshotsEqual, diffSnapshots } from '../utils/stateSnapshots';

/**
 * Extended parity harness for the known seed-5 backend vs sandbox mismatch.
 *
 * This test replays the full sandbox AI trace for square8 / 2p / seed=5 into
 * both backend and sandbox engines and, after each canonical move, compares:
 *
 *   1) The JSON-serialisable GameState snapshot used by other parity tests.
 *   2) A normalised view of host-internal metadata that does not live on
 *      GameState but materially affects phase scheduling and victory logic:
 *        - Per-turn flags: hasPlacedThisTurn / mustMoveFromStackKey.
 *        - Decision-phase flags: pendingTerritorySelfElimination /
 *          pendingLineRewardElimination.
 *        - Last-player-standing tracking: LPS round index, current round
 *          first player, exclusive candidate for completed rounds, and the
 *          current round's actor mask.
 *   3) A **semantic** LPS snapshot that ignores raw counters and focuses on:
 *        - For each player: hasAnyRealAction + hasMaterial.
 *        - Which player (if any) is the exclusive R172 candidate.
 *
 * It then logs the earliest index where:
 *   - GameState diverges,
 *   - raw internal metadata diverges, and
 *   - semantic LPS invariants diverge **while GameState is still equal**.
 *
 * This harness is intentionally DIAGNOSTIC-ONLY for now: it does not fail
 * when mismatches are found, so that it can be used while the underlying
 * bugs are being iteratively fixed.
 */

type InternalStateSnapshot = {
  hasPlacedThisTurn: boolean;
  mustMoveFromStackKey: string | null;
  pendingTerritorySelfElimination: boolean;
  pendingLineRewardElimination: boolean;
  lpsRoundIndex: number | null;
  lpsCurrentRoundFirstPlayer: number | null;
  lpsExclusivePlayerForCompletedRound: number | null;
  lpsCurrentRoundActorMask: Array<{ playerNumber: number; hadRealAction: boolean }>;
};

// Semantic LPS snapshot focused on rules-level invariants rather than
// raw host-internal counters.
interface SemanticLpsSnapshot {
  perPlayer: Array<{
    playerNumber: number;
    hasRealAction: boolean;
    hasMaterial: boolean;
  }>;
  exclusiveCandidate: number | null;
}

function normaliseBackendInternalState(engine: any): InternalStateSnapshot {
  const mask: Array<{ playerNumber: number; hadRealAction: boolean }> = [];
  // Access via lpsState (new shared structure)
  const lpsState = engine.lpsState;
  const rawMask: Map<number, boolean> | undefined = lpsState?.currentRoundActorMask;

  if (rawMask && rawMask instanceof Map) {
    for (const [playerNumber, hadRealAction] of rawMask.entries()) {
      mask.push({ playerNumber, hadRealAction: hadRealAction === true });
    }
    mask.sort((a, b) => a.playerNumber - b.playerNumber);
  }

  return {
    hasPlacedThisTurn: engine.hasPlacedThisTurn === true,
    mustMoveFromStackKey: engine.mustMoveFromStackKey ?? null,
    pendingTerritorySelfElimination: engine.pendingTerritorySelfElimination === true,
    pendingLineRewardElimination: engine.pendingLineRewardElimination === true,
    lpsRoundIndex:
      typeof lpsState?.roundIndex === 'number' && Number.isFinite(lpsState.roundIndex)
        ? lpsState.roundIndex
        : null,
    lpsCurrentRoundFirstPlayer:
      typeof lpsState?.currentRoundFirstPlayer === 'number'
        ? lpsState.currentRoundFirstPlayer
        : null,
    lpsExclusivePlayerForCompletedRound:
      typeof lpsState?.exclusivePlayerForCompletedRound === 'number'
        ? lpsState.exclusivePlayerForCompletedRound
        : null,
    lpsCurrentRoundActorMask: mask,
  };
}

function normaliseSandboxInternalState(engine: any): InternalStateSnapshot {
  const mask: Array<{ playerNumber: number; hadRealAction: boolean }> = [];
  // Access via _lpsState (new shared structure)
  const lpsState = engine._lpsState;
  const rawMask: Map<number, boolean> | undefined = lpsState?.currentRoundActorMask;

  if (rawMask && rawMask instanceof Map) {
    for (const [playerNumber, hadRealAction] of rawMask.entries()) {
      mask.push({ playerNumber, hadRealAction: hadRealAction === true });
    }
    mask.sort((a, b) => a.playerNumber - b.playerNumber);
  }

  return {
    hasPlacedThisTurn: engine._hasPlacedThisTurn === true,
    mustMoveFromStackKey: engine._mustMoveFromStackKey ?? null,
    pendingTerritorySelfElimination: engine._pendingTerritorySelfElimination === true,
    // Sandbox currently tracks line-reward eliminations under
    // _pendingLineRewardElimination.
    pendingLineRewardElimination: engine._pendingLineRewardElimination === true,
    lpsRoundIndex:
      typeof lpsState?.roundIndex === 'number' && Number.isFinite(lpsState.roundIndex)
        ? lpsState.roundIndex
        : null,
    lpsCurrentRoundFirstPlayer:
      typeof lpsState?.currentRoundFirstPlayer === 'number'
        ? lpsState.currentRoundFirstPlayer
        : null,
    lpsExclusivePlayerForCompletedRound:
      typeof lpsState?.exclusivePlayerForCompletedRound === 'number'
        ? lpsState.exclusivePlayerForCompletedRound
        : null,
    lpsCurrentRoundActorMask: mask,
  };
}

function getBackendSemanticLpsSnapshot(engine: any, state: GameState): SemanticLpsSnapshot {
  const perPlayer = state.players
    .slice()
    .sort((a, b) => a.playerNumber - b.playerNumber)
    .map((p) => {
      const playerNumber = p.playerNumber;
      const hasMaterial =
        typeof engine.playerHasMaterial === 'function'
          ? engine.playerHasMaterial(playerNumber) === true
          : false;
      const hasRealAction =
        typeof engine.hasAnyRealActionForPlayer === 'function'
          ? engine.hasAnyRealActionForPlayer(state, playerNumber) === true
          : false;
      return { playerNumber, hasRealAction, hasMaterial };
    });

  const lpsState = engine.lpsState;
  const exclusiveCandidate =
    typeof lpsState?.exclusivePlayerForCompletedRound === 'number'
      ? (lpsState.exclusivePlayerForCompletedRound as number)
      : null;

  return { perPlayer, exclusiveCandidate };
}

function getSandboxSemanticLpsSnapshot(engine: any): SemanticLpsSnapshot {
  const state: GameState = engine.getGameState();

  const perPlayer = state.players
    .slice()
    .sort((a, b) => a.playerNumber - b.playerNumber)
    .map((p) => {
      const playerNumber = p.playerNumber;
      const hasMaterial =
        typeof engine.playerHasMaterial === 'function'
          ? engine.playerHasMaterial(playerNumber) === true
          : false;
      const hasRealAction =
        typeof engine.hasAnyRealActionForPlayer === 'function'
          ? engine.hasAnyRealActionForPlayer(playerNumber) === true
          : false;
      return { playerNumber, hasRealAction, hasMaterial };
    });

  const lpsState = engine._lpsState;
  const exclusiveCandidate =
    typeof lpsState?.exclusivePlayerForCompletedRound === 'number'
      ? (lpsState.exclusivePlayerForCompletedRound as number)
      : null;

  return { perPlayer, exclusiveCandidate };
}

function createSandboxEngineFromInitial(initial: GameState): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType: initial.boardType,
    numPlayers: initial.players.length,
    playerKinds: initial.players
      .slice()
      .sort((a, b) => a.playerNumber - b.playerNumber)
      .map((p) => p.type as 'human' | 'ai'),
  };

  const handler: SandboxInteractionHandler = {
    async requestChoice(choice: any) {
      const options = ((choice as any).options as any[]) ?? [];
      const selectedOption = options.length > 0 ? options[0] : undefined;

      return {
        choiceId: (choice as any).id,
        playerNumber: (choice as any).playerNumber,
        choiceType: (choice as any).type,
        selectedOption,
      } as any;
    },
  };

  const engine = new ClientSandboxEngine({
    config,
    interactionHandler: handler,
    traceMode: true,
  });

  // Seed with the exact initial state from the trace so both hosts start
  // from identical geometry and counters.
  const engineAny: any = engine;
  engineAny.gameState = initial;
  return engine;
}

// This harness is sandbox-internal and relies on ClientSandboxEngine traceMode
// metadata. It is skipped by default and should be enabled only for targeted
// adapter debugging while sandbox coercions are still present.
const enableSandboxInternalParity = process.env.RINGRIFT_ENABLE_SANDBOX_INTERNAL_PARITY === '1';
const maybeDescribe = enableSandboxInternalParity ? describe : describe.skip;

maybeDescribe(
  'Backend vs Sandbox internal-state parity diagnostics (square8 / 2p / seed=5)',
  () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const seed = 5;
    const MAX_STEPS = 80; // generous upper bound; actual game ends earlier

    test('logs earliest divergence in GameState and internal metadata across full seed-5 trace', async () => {
      const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
      expect(trace.entries.length).toBeGreaterThan(0);

      const moves: Move[] = trace.entries.map((e) => e.action as Move);

      const backendEngine = createBackendEngineFromInitialState(trace.initialState);
      const sandboxEngine = createSandboxEngineFromInitial(trace.initialState);

      // Any-cast views used for invariant helpers and internal LPS/material
      // checks without widening the public engine surface.
      const backendAny: any = backendEngine;
      const sandboxAny: any = sandboxEngine;

      let firstGameStateMismatchIndex = -1;
      let firstInternalMismatchIndex = -1;
      let firstSemanticLpsMismatchIndex = -1;

      for (let i = 0; i < moves.length; i++) {
        const move = moves[i];

        // --- Apply move i to backend ---
        const backendStateBefore = backendEngine.getGameState();
        const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
        const matching = findMatchingBackendMove(move, backendValidMoves);

        if (!matching) {
          console.error('[Seed5 InternalParity] No matching backend move', {
            index: i,
            moveNumber: move.moveNumber,
            type: move.type,
            player: move.player,
            backendCurrentPlayer: backendStateBefore.currentPlayer,
            backendCurrentPhase: backendStateBefore.currentPhase,
            backendValidMovesCount: backendValidMoves.length,
          });
          firstGameStateMismatchIndex = i;
          break;
        }

        const { id, timestamp, moveNumber, ...payload } = matching as any;
        const backendResult = await backendEngine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        if (!backendResult.success) {
          console.error('[Seed5 InternalParity] Backend makeMove failed', {
            index: i,
            moveNumber: move.moveNumber,
            type: move.type,
            player: move.player,
            backendMoveNumber: (matching as any).moveNumber,
            error: backendResult.error,
          });
          firstGameStateMismatchIndex = i;
          break;
        }

        // Step through automatic phases after each move so backend reaches the
        // same stopping point as the sandbox (which fully processes its
        // post-move consequences inside applyCanonicalMove / advanceAfterMovement).
        await backendEngine.stepAutomaticPhasesForTesting();

        // --- Apply move i to sandbox ---
        await sandboxEngine.applyCanonicalMove(move);

        const backendAfter = backendEngine.getGameState();
        const sandboxAfter = sandboxEngine.getGameState();

        const backendSnap = snapshotFromGameState(`backend-step-${i}`, backendAfter);
        const sandboxSnap = snapshotFromGameState(`sandbox-step-${i}`, sandboxAfter);

        const backendInternal = normaliseBackendInternalState(backendEngine as any);
        const sandboxInternal = normaliseSandboxInternalState(sandboxEngine as any);

        // Invariant-style diagnostics: detect "active but no real actions" states
        // for each host. This mirrors the Python invariants used in the rules
        // service and helps pinpoint where forced elimination / skipping should
        // have triggered but did not.
        const backendHasMaterialAndNoRealAction = (() => {
          if (backendAfter.gameStatus !== 'active') {
            return false;
          }
          if (
            typeof backendAny.playerHasMaterial !== 'function' ||
            typeof backendAny.hasAnyRealActionForPlayer !== 'function'
          ) {
            return false;
          }
          const p = backendAfter.currentPlayer;
          const hasMaterial = backendAny.playerHasMaterial(p) === true;
          const hasReal = backendAny.hasAnyRealActionForPlayer(backendAfter, p) === true;
          return hasMaterial && !hasReal;
        })();

        const sandboxHasMaterialAndNoRealAction = (() => {
          if (sandboxAfter.gameStatus !== 'active') {
            return false;
          }
          if (
            typeof sandboxAny.playerHasMaterial !== 'function' ||
            typeof sandboxAny.hasAnyRealActionForPlayer !== 'function'
          ) {
            return false;
          }
          const p = sandboxAfter.currentPlayer;
          const hasMaterial = sandboxAny.playerHasMaterial(p) === true;
          const hasReal = sandboxAny.hasAnyRealActionForPlayer(p) === true;
          return hasMaterial && !hasReal;
        })();

        if (backendHasMaterialAndNoRealAction || sandboxHasMaterialAndNoRealAction) {
          console.warn('[Seed5 InternalParity] ACTIVE_NO_MOVES invariant hit', {
            index: i,
            moveNumber: move.moveNumber,
            type: move.type,
            player: move.player,
            backend: {
              currentPlayer: backendAfter.currentPlayer,
              currentPhase: backendAfter.currentPhase,
              gameStatus: backendAfter.gameStatus,
              hasMaterialAndNoRealAction: backendHasMaterialAndNoRealAction,
            },
            sandbox: {
              currentPlayer: sandboxAfter.currentPlayer,
              currentPhase: sandboxAfter.currentPhase,
              gameStatus: sandboxAfter.gameStatus,
              hasMaterialAndNoRealAction: sandboxHasMaterialAndNoRealAction,
            },
          });
        }

        const backendSemanticLps = getBackendSemanticLpsSnapshot(
          backendEngine as any,
          backendAfter
        );
        const sandboxSemanticLps = getSandboxSemanticLpsSnapshot(sandboxEngine as any);

        const gameStatesEqual = snapshotsEqual(backendSnap, sandboxSnap);
        const internalsEqual = JSON.stringify(backendInternal) === JSON.stringify(sandboxInternal);
        const semanticLpsEqual =
          JSON.stringify(backendSemanticLps) === JSON.stringify(sandboxSemanticLps);

        // Around the late-game tail, always log a compact view even if equal.
        if (i >= 60 && i <= 64) {
          // eslint-disable-next-line no-console
          console.log('[Seed5 InternalParity] tail window snapshot', {
            index: i,
            moveNumber: move.moveNumber,
            type: move.type,
            player: move.player,
            backend: {
              currentPlayer: backendAfter.currentPlayer,
              currentPhase: backendAfter.currentPhase,
              gameStatus: backendAfter.gameStatus,
              winner: backendAfter.winner,
              internal: backendInternal,
              semanticLps: backendSemanticLps,
            },
            sandbox: {
              currentPlayer: sandboxAfter.currentPlayer,
              currentPhase: sandboxAfter.currentPhase,
              gameStatus: sandboxAfter.gameStatus,
              winner: sandboxAfter.winner,
              internal: sandboxInternal,
              semanticLps: sandboxSemanticLps,
            },
            gameStatesEqual,
            internalsEqual,
            semanticLpsEqual,
          });
        }

        if (!gameStatesEqual && firstGameStateMismatchIndex === -1) {
          firstGameStateMismatchIndex = i;
          const diff = diffSnapshots(backendSnap, sandboxSnap);

          console.error('[Seed5 InternalParity] FIRST GameState mismatch', {
            index: i,
            moveNumber: move.moveNumber,
            type: move.type,
            player: move.player,
            diff,
          });
        }

        if (!internalsEqual && firstInternalMismatchIndex === -1) {
          firstInternalMismatchIndex = i;

          console.error('[Seed5 InternalParity] FIRST internal-state mismatch', {
            index: i,
            moveNumber: move.moveNumber,
            type: move.type,
            player: move.player,
            backendInternal,
            sandboxInternal,
          });
        }

        if (!semanticLpsEqual && firstSemanticLpsMismatchIndex === -1 && gameStatesEqual) {
          firstSemanticLpsMismatchIndex = i;

          console.error(
            '[Seed5 InternalParity] FIRST semantic LPS mismatch (with equal GameState)',
            {
              index: i,
              moveNumber: move.moveNumber,
              type: move.type,
              player: move.player,
              backendSemanticLps,
              sandboxSemanticLps,
            }
          );
        }
      }

      // eslint-disable-next-line no-console
      console.log('[Seed5 InternalParity] summary', {
        seed,
        totalMoves: moves.length,
        firstGameStateMismatchIndex,
        firstInternalMismatchIndex,
        firstSemanticLpsMismatchIndex,
      });

      // Diagnostic harness: ensure we at least exercised the trace.
      expect(moves.length).toBeGreaterThan(0);
    });
  }
);
