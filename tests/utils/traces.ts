import {
  BoardType,
  BOARD_CONFIGS,
  GameState,
  GameTrace,
  Move,
  Player,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { GameEngine } from '../../src/server/game/GameEngine';
import { findMatchingBackendMove, describeMovesListForLog } from './moveMatching';
import { logAiDiagnostic } from './aiTestLogger';
import { formatMove, formatMoveList } from '../../src/shared/engine/notation';
import { computeProgressSnapshot, hashGameState } from '../../src/shared/engine/core';

const TRACE_DEBUG_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

/**
 * Tiny deterministic PRNG (same LCG as other AI simulation tests) so traces
 * can be reproduced by seed.
 */
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    // LCG parameters from Numerical Recipes
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/**
 * Deterministic SandboxInteractionHandler used for trace generation and
 * replay. In particular, capture_direction choices are resolved in a
 * stable, geometry-based way so traces are reproducible across runs.
 */
function createDeterministicSandboxHandler(): SandboxInteractionHandler {
  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice extends PlayerChoice>(
      choice: TChoice
    ): Promise<PlayerChoiceResponseFor<TChoice>> {
      const anyChoice = choice as any;

      if (anyChoice.type === 'capture_direction') {
        const cd = anyChoice as CaptureDirectionChoice;
        const options = cd.options || [];
        if (options.length === 0) {
          throw new Error('SandboxInteractionHandler: no options for capture_direction');
        }

        // Deterministically pick the option with the smallest landing x,y
        // to keep simulations reproducible given a fixed Math.random.
        let selected = options[0];
        for (const opt of options) {
          if (
            opt.landingPosition.x < selected.landingPosition.x ||
            (opt.landingPosition.x === selected.landingPosition.x &&
              opt.landingPosition.y < selected.landingPosition.y)
          ) {
            selected = opt;
          }
        }

        return {
          choiceId: cd.id,
          playerNumber: cd.playerNumber,
          choiceType: cd.type,
          selectedOption: selected,
        } as PlayerChoiceResponseFor<TChoice>;
      }

      const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption,
      } as PlayerChoiceResponseFor<TChoice>;
    },
  };

  return handler;
}

/**
 * Run a sandbox AI-vs-AI game under a seeded PRNG and return a GameTrace
 * containing the initial state and the structured history emitted by
 * ClientSandboxEngine. This is the primary building block for
 * sandbox-vs-backend parity and S-invariant debugging.
 */
export async function runSandboxAITrace(
  boardType: BoardType,
  numPlayers: number,
  seed: number,
  maxSteps: number
): Promise<GameTrace> {
  const rng = makePrng(seed);
  const originalRandom = Math.random;
  Math.random = rng;

  try {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
    };

    const handler = createDeterministicSandboxHandler();
    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });

    const initialState = engine.getGameState();

    for (let step = 0; step < maxSteps; step++) {
      if (TRACE_DEBUG_ENABLED) {
        // eslint-disable-next-line no-console
        console.log(`[runSandboxAITrace] Step ${step} start`);
      }
      const state = engine.getGameState();
      // DIAGNOSTIC: trace harness view of current player/phase before AI turn
      if (TRACE_DEBUG_ENABLED) {
        // eslint-disable-next-line no-console
        console.log('[runSandboxAITrace] State before maybeRunAITurn', {
          step,
          currentPlayer: state.currentPlayer,
          currentPhase: state.currentPhase,
          gameStatus: state.gameStatus,
          ringsInHand: state.players.find((p) => p.playerNumber === state.currentPlayer)
            ?.ringsInHand,
          stacksOnBoard: state.board.stacks.size,
        });
      }
      if (state.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!currentPlayer || currentPlayer.type !== 'ai') {
        // Non-AI to move (should be rare in this harness). Just break; the
        // trace up to this point is still useful for debugging.
        break;
      }

      // Use the same seeded RNG for sandbox AI decisions that the harness
      // uses for any global Math.random-based behaviour.
      await engine.maybeRunAITurn(rng);
    }

    const finalState = engine.getGameState();

    if (TRACE_DEBUG_ENABLED) {
      const initialProgress = computeProgressSnapshot(initialState);
      const initialHash = hashGameState(initialState);

      const firstEntries = finalState.history.slice(0, 5).map((entry) => ({
        moveNumber: entry.moveNumber,
        actor: entry.actor,
        phaseBefore: entry.phaseBefore,
        phaseAfter: entry.phaseAfter,
        statusBefore: entry.statusBefore,
        statusAfter: entry.statusAfter,
        S_before: entry.progressBefore?.S,
        S_after: entry.progressAfter?.S,
        notation: formatMove(entry.action, { boardType }),
      }));

      logAiDiagnostic(
        'sandbox-trace-opening-sequence',
        {
          boardType,
          numPlayers,
          seed,
          maxSteps,
          initial: {
            currentPlayer: initialState.currentPlayer,
            currentPhase: initialState.currentPhase,
            gameStatus: initialState.gameStatus,
            S: initialProgress.S,
            stateHash: initialHash,
          },
          historyLength: finalState.history.length,
          firstEntries,
        },
        'trace-parity'
      );
    }

    return {
      initialState,
      entries: finalState.history,
    };
  } finally {
    Math.random = originalRandom;
  }
}

/**
 * Construct a backend GameEngine from a trace's initial state. The engine
 * starts from an empty board with the same boardType, timeControl, and
 * player seats; canonical moves from the trace are then applied to reach
 * comparable positions for parity debugging.
 */
export function createBackendEngineFromInitialState(initial: GameState): GameEngine {
  const timeControl = initial.timeControl;
  const boardType = initial.boardType;
  const boardConfig = BOARD_CONFIGS[boardType];

  // Seed players with the same ordering and basic counters. GameEngine will
  // assign playerNumber sequentially; we rely on the initial ordering to
  // match the trace.
  const players: Player[] = initial.players
    .slice()
    .sort((a, b) => a.playerNumber - b.playerNumber)
    .map(
      (p) =>
        ({
          id: p.id || `trace-p${p.playerNumber}`,
          username: p.username || `Player ${p.playerNumber}`,
          type: p.type,
          playerNumber: p.playerNumber,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: boardConfig.ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        }) as Player
    );

  const engine = new GameEngine('trace-backend-replay', boardType, players, timeControl, false);
  const started = engine.startGame();
  if (!started) {
    throw new Error('Failed to start GameEngine for trace replay');
  }

  // For trace/parity harnesses, enable Move-driven decision phases so that
  // line and territory processing are expressed as explicit canonical Moves
  // (process_line, choose_line_reward, process_territory_region,
  // eliminate_rings_from_stack) instead of being resolved purely via
  // processAutomaticConsequences. This keeps backend behaviour aligned with
  // the sandbox trace model, which records these decisions as distinct
  // GameHistoryEntry actions.
  engine.enableMoveDrivenDecisionPhases();

  return engine;
}

/**
 * Replay a sequence of moves onto a fresh backend GameEngine constructed from
 * the provided initial state.
 *
 * This is useful for verifying backend behavior against a known sequence of
 * moves (e.g. from a sandbox trace or a manual test case).
 */
export async function replayMovesOnBackend(
  initialState: GameState,
  moves: Move[]
): Promise<GameTrace> {
  const engine = createBackendEngineFromInitialState(initialState);
  const backendInitialState = engine.getGameState();

  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];
    const nextMove: Move | undefined = moves[i + 1];

    // For backend replay we always advance from the backend's current
    // state.
    const backendStateBefore = engine.getGameState();
    const backendMoves = engine.getValidMoves(backendStateBefore.currentPlayer);

    const matchingBackendMove = findMatchingBackendMove(move, backendMoves);
    if (!matchingBackendMove) {
      if (TRACE_DEBUG_ENABLED) {
        const backendProgress = computeProgressSnapshot(backendStateBefore);
        const backendHash = hashGameState(backendStateBefore);

        logAiDiagnostic(
          'trace-parity-backend-move-mismatch',
          {
            sandboxMove: {
              raw: {
                type: move.type,
                player: move.player,
                from: move.from,
                to: move.to,
                captureTarget: move.captureTarget,
                moveNumber: move.moveNumber,
              },
              notation: formatMove(move, { boardType: backendStateBefore.boardType }),
            },
            backendStateBefore: {
              boardType: backendStateBefore.boardType,
              currentPlayer: backendStateBefore.currentPlayer,
              currentPhase: backendStateBefore.currentPhase,
              gameStatus: backendStateBefore.gameStatus,
              S: backendProgress.S,
              stateHash: backendHash,
              players: backendStateBefore.players.map((p: Player) => ({
                playerNumber: p.playerNumber,
                type: p.type,
                ringsInHand: p.ringsInHand,
                eliminatedRings: p.eliminatedRings,
                territorySpaces: p.territorySpaces,
              })),
              validMovesCount: backendMoves.length,
              validMovesNotation: formatMoveList(backendMoves as Move[], {
                boardType: backendStateBefore.boardType,
              }),
            },
          },
          'trace-parity'
        );
      }

      throw new Error(
        `replayMovesOnBackend: no matching backend move found for move ` +
          `moveNumber=${move.moveNumber}, player=${move.player}, ` +
          `move=${JSON.stringify({ type: move.type, from: move.from, to: move.to, captureTarget: move.captureTarget })}, ` +
          `backendMovesCount=${backendMoves.length}\n` +
          describeMovesListForLog(backendMoves)
      );
    }

    const { id, timestamp, moveNumber, ...payload } = matchingBackendMove;
    const result = await engine.makeMove(payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
    if (!result.success) {
      throw new Error(
        `replayMovesOnBackend: makeMove failed at backend moveNumber=${matchingBackendMove.moveNumber}: ${result.error}`
      );
    }

    // NOTE: Earlier versions of the trace harness automatically resolved
    // backend chain_capture continuations here when sandbox traces collapsed
    // entire chains into a single overtaking_capture move. Now that sandbox
    // traces emit one canonical move per capture segment (including explicit
    // continue_capture_segment actions), we rely solely on the explicit move
    // list and no longer auto-resolve backend continuations.
  }

  const finalState = engine.getGameState();
  return {
    initialState: backendInitialState,
    entries: finalState.history,
  };
}

/**
 * When the backend GameEngine is in the chain_capture phase with at least
 * one legal continuation segment, automatically apply follow-up capture
 * segments until the chain is exhausted.
 *
 * This mirrors the deterministic capture_direction behaviour used by the
 * sandbox trace harness: when multiple options are available, always pick
 * the lexicographically smallest landing position (x,y,z). If the next
 * sandbox move is already a continue_capture_segment for the active player,
 * we skip auto-resolution and rely on the explicit trace entry instead.
 */
async function autoResolveChainCaptureIfNeeded(
  engine: GameEngine,
  nextSandboxMove: Move | undefined
): Promise<void> {
  // First step through any internal bookkeeping phases so we only ever
  // inspect interactive phases.
  engine.stepAutomaticPhasesForTesting();

  // Resolve at most a bounded number of segments defensively; in
  // well-formed states the chain must eventually terminate.
  const MAX_SEGMENTS = 32;

  for (let i = 0; i < MAX_SEGMENTS; i++) {
    const state = engine.getGameState();
    if (state.gameStatus !== 'active' || state.currentPhase !== 'chain_capture') {
      return;
    }

    const player = state.currentPlayer;

    // If the sandbox trace already provides an explicit
    // continue_capture_segment from this chain position for this player,
    // do not auto-resolve; the next replay loop iteration will apply it.
    if (
      nextSandboxMove &&
      nextSandboxMove.type === 'continue_capture_segment' &&
      nextSandboxMove.player === player
    ) {
      return;
    }

    const moves = engine.getValidMoves(player);
    const continuations = moves.filter((m) => m.type === 'continue_capture_segment');

    if (continuations.length === 0) {
      // Defensive: clear any stale chain state by advancing through
      // automatic phases; callers will observe the resolved state.
      engine.stepAutomaticPhasesForTesting();
      return;
    }

    // Deterministically select the continuation with the lexicographically
    // smallest landing position. This keeps behaviour in sync with the
    // sandbox capture_direction handler used during trace generation.
    const chosen = continuations.reduce((best, current) => {
      const bz = (best.to.z ?? 0) as number;
      const cz = (current.to.z ?? 0) as number;

      if (current.to.x < best.to.x) return current;
      if (current.to.x > best.to.x) return best;
      if (current.to.y < best.to.y) return current;
      if (current.to.y > best.to.y) return best;
      if (cz < bz) return current;
      if (cz > bz) return best;
      return best;
    }, continuations[0]);

    const { id, timestamp, moveNumber, ...payload } = chosen;
    const result = await engine.makeMove(payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
    if (!result.success) {
      throw new Error(
        `autoResolveChainCaptureIfNeeded: makeMove failed at backend moveNumber=${chosen.moveNumber}: ${result.error}`
      );
    }

    // Loop to see if additional chain_capture segments remain.
    engine.stepAutomaticPhasesForTesting();
  }

  // Safety net: if we somehow performed MAX_SEGMENTS continuations and are
  // still in an active chain_capture phase, leave further resolution to the
  // caller rather than risk an infinite loop.
}

/**
 * Replay a GameTrace onto a fresh backend GameEngine constructed from the
 * trace's initial configuration. Returns the backend's own GameTrace
 * (initial state + history entries) so tests can compare S metrics,
 * state hashes, and summaries step-by-step against the original trace.
 */
export async function replayTraceOnBackend(trace: GameTrace): Promise<GameTrace> {
  return replayMovesOnBackend(
    trace.initialState,
    trace.entries.map((e) => e.action)
  );
}

/**
 * Construct a fresh sandbox engine from a trace's initial state. This uses
 * the same boardType and player count; all players are treated as AI for
 * replay purposes.
 */
function createSandboxEngineFromInitialState(initial: GameState): ClientSandboxEngine {
  const boardType = initial.boardType;
  const numPlayers = initial.players.length;

  const config: SandboxConfig = {
    boardType,
    numPlayers,
    playerKinds: initial.players
      .slice()
      .sort((a, b) => a.playerNumber - b.playerNumber)
      .map((p) => p.type as 'human' | 'ai'),
  };

  const handler = createDeterministicSandboxHandler();
  return new ClientSandboxEngine({
    config,
    interactionHandler: handler,
    // Trace replays are always run under the parity-focused trace
    // harness, so enable traceMode explicitly here as well.
    traceMode: true,
  });
}

/**
 * Replay a GameTrace's canonical moves onto a fresh ClientSandboxEngine.
 * This is primarily used to:
 *   - verify that a sandbox trace is deterministic when re-applied, and
 *   - compare sandbox-vs-backend behaviour for the same action list.
 */
export async function replayTraceOnSandbox(trace: GameTrace): Promise<GameTrace> {
  const engine = createSandboxEngineFromInitialState(trace.initialState);
  const initialState = engine.getGameState();

  for (const entry of trace.entries) {
    const move = entry.action;
    await engine.applyCanonicalMove(move);
  }

  const finalState = engine.getGameState();
  return {
    initialState,
    entries: finalState.history,
  };
}
