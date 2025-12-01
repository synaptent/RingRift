import { GameEngine } from '../../src/server/game/GameEngine';
import { GameState, Player, TimeControl, BOARD_CONFIGS } from '../../src/shared/types/game';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';
import { logAiDiagnostic } from '../utils/aiTestLogger';

// Simple LCG for deterministic testing without external dependencies
function createLCG(seed: number) {
  return function () {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

// Mock the AI service client to simulate downtime/failure
jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: () => ({
    getAIMove: jest.fn().mockRejectedValue(new Error('Service unavailable')),
    getLineRewardChoice: jest.fn().mockRejectedValue(new Error('Service unavailable')),
    getRingEliminationChoice: jest.fn().mockRejectedValue(new Error('Service unavailable')),
    getRegionOrderChoice: jest.fn().mockRejectedValue(new Error('Service unavailable')),
  }),
}));

/**
 * SKIP REASON: Heavy AI vs AI soak test - not suitable for regular CI
 *
 * This test runs a complete AI vs AI game with:
 * - 4000 max moves limit
 * - 30 second timeout
 * - All game phases (placement, movement, capture, chain capture, line, territory)
 *
 * Why skipped:
 * 1. Too slow for CI (can take 30+ seconds)
 * 2. Previously timed out or stalled due to S-invariant plateau detection
 * 3. Functionally equivalent to scripts/run-orchestrator-soak.ts which is run
 *    nightly via .github/workflows/orchestrator-soak-nightly.yml
 *
 * To run manually for debugging:
 *   npm test -- --testPathPattern="FullGameFlow" --no-coverage
 *   (Remember to remove describe.skip first)
 */
describe.skip('Full Game Flow Integration (AI Fallback)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  // Allow for long AI-vs-AI sequences; moves here are individual actions rather
  // than full turns, so a realistic game may require well over 500 moves.
  // We keep a hard cap to avoid pathological infinite loops in case of
  // regressions.
  const MAX_MOVES = 4000;

  it('completes a full game using local AI fallback when service is down', async () => {
    // Use a fixed seed for reproducibility
    // Using a numeric seed derived from the string 'ringrift-fallback-test-seed'
    const rng = createLCG(123456789);

    // Setup 2 AI players
    const players: Player[] = [
      {
        id: 'p1',
        username: 'AI-1',
        playerNumber: 1,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'AI-2',
        playerNumber: 2,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    // Configure AI engine for these players
    globalAIEngine.createAI(1, 5);
    globalAIEngine.createAI(2, 5);

    const engine = new GameEngine('integration-test', 'square8', players, timeControl);

    engine.startGame();

    // Diagnostic S-invariant and stall tracking. This mirrors the AI simulation
    // harness: we track the progress snapshot S = markers + collapsedSpaces +
    // eliminated and log periodic summaries so long-running games can be
    // understood when they approach the MAX_MOVES cap.
    const initialState: GameState = engine.getGameState();
    let lastProgress = computeProgressSnapshot(initialState);
    let lastSChangeMove = 0;
    const STALL_WINDOW = 200;
    const LOG_INTERVAL = 5;
    let stalled = false;

    let moves = 0;
    while (engine.getGameState().gameStatus === 'active' && moves < MAX_MOVES) {
      const state = engine.getGameState();
      const progress = computeProgressSnapshot(state);

      const diagnosticBase = {
        moves,
        currentPlayer: state.currentPlayer,
        currentPhase: state.currentPhase,
        gameStatus: state.gameStatus,
        S: progress.S,
        markers: progress.markers,
        collapsed: progress.collapsed,
        eliminated: progress.eliminated,
      };

      if (moves % LOG_INTERVAL === 0) {
        logAiDiagnostic(
          'full-game-flow-summary',
          {
            ...diagnosticBase,
            players: state.players.map((p) => ({
              playerNumber: p.playerNumber,
              type: p.type,
              ringsInHand: p.ringsInHand,
              eliminatedRings: p.eliminatedRings,
              territorySpaces: p.territorySpaces,
              stacks: Array.from(state.board.stacks.values()).filter(
                (s) => s.controllingPlayer === p.playerNumber
              ).length,
            })),
          },
          'full-game-flow'
        );
      }

      if (progress.S !== lastProgress.S) {
        // Log every S-invariant change so plateaus and regressions can be
        // correlated with concrete moves in the saved diagnostic logs.
        logAiDiagnostic(
          'S-invariant-change',
          {
            ...diagnosticBase,
            lastProgress,
            delta: {
              dS: progress.S - lastProgress.S,
              dMarkers: progress.markers - lastProgress.markers,
              dCollapsed: progress.collapsed - lastProgress.collapsed,
              dEliminated: progress.eliminated - lastProgress.eliminated,
            },
          },
          'full-game-flow'
        );

        // Defensive check: S should never decrease under the compact rules.
        if (progress.S < lastProgress.S) {
          logAiDiagnostic(
            'S-invariant-decreased',
            {
              ...diagnosticBase,
              lastProgress,
            },
            'full-game-flow'
          );
          throw new Error(
            `[FullGameFlow] S-invariant decreased from ${lastProgress.S} to ${progress.S} at move ${moves}`
          );
        }

        lastProgress = progress;
        lastSChangeMove = moves;
      } else {
        // Log every plateau step so long stalls can be reconstructed from
        // the diagnostic log without relying on console output.
        let validMovesSummary: {
          count: number;
          types: Record<string, number>;
        } | null = null;

        if (
          state.currentPhase === 'ring_placement' ||
          state.currentPhase === 'movement' ||
          state.currentPhase === 'capture' ||
          state.currentPhase === 'chain_capture'
        ) {
          const validMovesForCurrentPlayer = engine.getValidMoves(state.currentPlayer);
          const typeCounts: Record<string, number> = {};
          for (const m of validMovesForCurrentPlayer) {
            typeCounts[m.type] = (typeCounts[m.type] ?? 0) + 1;
          }
          validMovesSummary = {
            count: validMovesForCurrentPlayer.length,
            types: typeCounts,
          };
        }

        logAiDiagnostic(
          'S-invariant-plateau-step',
          {
            ...diagnosticBase,
            lastSChangeMove,
            plateauLength: moves - lastSChangeMove,
            validMovesSummary,
          },
          'full-game-flow'
        );

        if (moves - lastSChangeMove >= STALL_WINDOW) {
          stalled = true;
          logAiDiagnostic(
            'S-invariant-stalled',
            {
              ...diagnosticBase,
              lastSChangeMove,
              plateauLength: moves - lastSChangeMove,
            },
            'full-game-flow'
          );
          break;
        }
      }

      // If it's an interactive phase, the AI engine should generate a move.
      // This now includes:
      //   - Core phases: ring_placement, movement, capture, chain_capture
      //   - Advanced decision phases: line_processing, territory_processing
      // In all of these, legal actions are exposed as canonical Move objects
      // via GameEngine.getValidMoves and must be chosen explicitly rather than
      // relying on bespoke PlayerChoice-only flows.
      const isInteractivePhase =
        state.currentPhase === 'ring_placement' ||
        state.currentPhase === 'movement' ||
        state.currentPhase === 'capture' ||
        state.currentPhase === 'chain_capture' ||
        state.currentPhase === 'line_processing' ||
        state.currentPhase === 'territory_processing';

      if (isInteractivePhase) {
        // In fallback mode we deliberately drive move selection from
        // GameEngine.getValidMoves so that termination behaviour matches the
        // dedicated backend AI simulation harness. AIEngine is still used for
        // local prioritisation among candidates across all interactive phases,
        // including advanced decision phases.
        const validMoves = engine.getValidMoves(state.currentPlayer);

        if (validMoves.length > 0) {
          const move = globalAIEngine.chooseLocalMoveFromCandidates(
            state.currentPlayer,
            state,
            validMoves,
            rng
          );

          if (!move) {
            // Extremely defensive: if the selector returns null despite having
            // candidates, treat this as a blocked state and fall through to
            // the same resolver used when there are no valid moves.
            engine.resolveBlockedStateForCurrentPlayerForTesting();
          } else {
            const { id, timestamp, moveNumber, ...payload } = move as any;
            const result = await engine.makeMove(payload);
            if (!result.success) {
              console.error('Move failed:', result, move);
            }
            expect(result.success).toBe(true);
          }
        } else {
          // No legal moves for the current player in an interactive phase:
          // attempt to resolve a blocked state using the same safety net the
          // AI simulation harness uses. This can apply forced elimination /
          // structural stalemate resolution before we give up on the loop.
          engine.resolveBlockedStateForCurrentPlayerForTesting();
        }
      } else {
        // Non-interactive bookkeeping phases (if any) are advanced without
        // explicit Move selection.
        engine.stepAutomaticPhasesForTesting();
      }

      moves++;
    }

    const finalState = engine.getGameState();
    logAiDiagnostic(
      'full-game-flow-final-state',
      {
        moves,
        finalStatus: finalState.gameStatus,
        finalPhase: finalState.currentPhase,
        finalPlayer: finalState.currentPlayer,
        finalProgress: computeProgressSnapshot(finalState),
      },
      'full-game-flow'
    );

    if (stalled) {
      throw new Error(
        `[FullGameFlow] S-invariant stalled for ${moves - lastSChangeMove} moves ` +
          `(S=${lastProgress.S}, status=${finalState.gameStatus}, ` +
          `phase=${finalState.currentPhase}, currentPlayer=${finalState.currentPlayer})`
      );
    }

    // Assert game finished naturally
    expect(finalState.gameStatus).not.toBe('active');
    expect(['completed', 'finished']).toContain(finalState.gameStatus);
  }, 30000); // Increase timeout for full game simulation
});
