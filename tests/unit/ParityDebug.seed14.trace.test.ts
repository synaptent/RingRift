import {
  runSandboxAITrace,
  replayMovesOnBackend,
  createBackendEngineFromInitialState,
} from '../utils/traces';
import { GameState, Move, Position, positionToString } from '../../src/shared/types/game';
import { summarizeBoard, hashGameState } from '../../src/shared/engine/core';
import {
  enumerateCaptureSegmentsFromBoard,
  CaptureBoardAdapters,
} from '../../src/client/sandbox/sandboxCaptures';
import { findMatchingBackendMove } from '../utils/moveMatching';

describe('Parity Debug: Seed 14 Trace', () => {
  test('square8 / 2p / seed=14: debug failure state (first mismatch)', async () => {
    const seed = 14;
    const boardType = 'square8';
    const numPlayers = 2;
    const maxSteps = 50;

    // 1. Run Sandbox Trace
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, maxSteps);

    // 2. Replay on Backend until failure
    try {
      await replayMovesOnBackend(
        trace.initialState,
        trace.entries.map((e) => e.action)
      );
    } catch (e: any) {
      console.log('Caught expected failure:', e.message);

      // Let's manually replay step-by-step using the same backend construction
      // logic as replayMovesOnBackend so that any divergence we see here
      // matches the real parity harness behaviour.
      const engine = createBackendEngineFromInitialState(trace.initialState as GameState);

      for (const entry of trace.entries) {
        const move = entry.action;
        engine.stepAutomaticPhasesForTesting();

        const backendStateBefore = engine.getGameState();
        const backendMoves = engine.getValidMoves(backendStateBefore.currentPlayer);

        const matching = findMatchingBackendMove(move as Move, backendMoves as Move[]);

        if (!matching) {
          console.log('FAILURE FOUND at move', move.moveNumber);
          console.log('Sandbox Move:', JSON.stringify(move, null, 2));

          // Also log the sandbox-side board summary from the original trace
          const traceEntry = trace.entries.find((e) => e.action.moveNumber === move.moveNumber);
          if (traceEntry) {
            console.log(
              'Sandbox State Summary BEFORE move (from trace history):',
              JSON.stringify(traceEntry.boardBeforeSummary, null, 2)
            );
            console.log('Sandbox State Hash (before move):', traceEntry.stateHashBefore);
            console.log(
              'Sandbox State Summary AFTER move (from trace history):',
              JSON.stringify(traceEntry.boardAfterSummary, null, 2)
            );
            console.log('Sandbox State Hash (after move):', traceEntry.stateHashAfter);
          }

          const backendState = engine.getGameState();
          console.log(
            'Backend State Summary:',
            JSON.stringify(summarizeBoard(backendState.board), null, 2)
          );
          console.log('Backend State Hash:', hashGameState(backendState));
          console.log('Backend Valid Moves:', JSON.stringify(backendMoves, null, 2));

          // Inspect specific stack details
          if (move.from) {
            const attackerKey = `${move.from.x},${move.from.y}`;
            const stack = backendState.board.stacks.get(attackerKey);
            console.log('Attacker Stack:', JSON.stringify(stack, null, 2));
          }
          if (move.captureTarget) {
            const targetKey = `${move.captureTarget.x},${move.captureTarget.y}`;
            const stack = backendState.board.stacks.get(targetKey);
            console.log('Target Stack:', JSON.stringify(stack, null, 2));
          }

          // Additionally, enumerate sandbox-style capture segments directly from the
          // backend board using the shared sandboxCaptures helper so we can see
          // exactly which overtaking_capture options the sandbox logic believes
          // are legal from this position.
          if (move.type === 'overtaking_capture' && move.from) {
            const backendBoard = backendState.board;

            const adapters: CaptureBoardAdapters = {
              isValidPosition: (pos: Position) => {
                // square8-specific bounds (0..7 in both axes)
                return (
                  pos.x >= 0 && pos.x < backendBoard.size && pos.y >= 0 && pos.y < backendBoard.size
                );
              },
              isCollapsedSpace: (pos: Position, board) => {
                const key = positionToString(pos);
                return board.collapsedSpaces.has(key);
              },
              getMarkerOwner: (pos: Position, board) => {
                const key = positionToString(pos);
                const marker = board.markers.get(key);
                return marker?.player;
              },
            };

            const segments = enumerateCaptureSegmentsFromBoard(
              'square8',
              backendBoard,
              move.from,
              move.player,
              adapters
            );

            console.log(
              'Sandbox-style capture segments from backend board:',
              JSON.stringify(segments, null, 2)
            );

            const matchingSegment = segments.find((seg) => {
              const sameFrom = seg.from.x === move.from!.x && seg.from.y === move.from!.y;
              const sameLanding = seg.landing.x === move.to!.x && seg.landing.y === move.to!.y;
              const sameTarget =
                !!move.captureTarget &&
                seg.target.x === move.captureTarget.x &&
                seg.target.y === move.captureTarget.y;
              return sameFrom && sameLanding && sameTarget;
            });

            console.log(
              'Matching sandbox-style capture segment for failing move:',
              JSON.stringify(matchingSegment ?? null, null, 2)
            );
          }

          break;
        }

        await engine.makeMove(move as Move);
      }
    }
  });

  test('square8 / 2p / seed=14: debug parity state at move 33', async () => {
    const seed = 14;
    const boardType = 'square8';
    const numPlayers = 2;
    const maxSteps = 60;
    const TARGET_MOVE = 33;

    const trace = await runSandboxAITrace(boardType, numPlayers, seed, maxSteps);

    const engine = createBackendEngineFromInitialState(trace.initialState as GameState);

    for (const entry of trace.entries) {
      const move = entry.action;
      engine.stepAutomaticPhasesForTesting();

      const backendStateBefore = engine.getGameState();
      const backendMoves = engine.getValidMoves(backendStateBefore.currentPlayer);

      const matching = findMatchingBackendMove(move as Move, backendMoves as Move[]);
      const isTargetMove = move.moveNumber === TARGET_MOVE;

      if (!matching || isTargetMove) {
        console.log('FAILURE OR TARGET STATE at move', move.moveNumber);
        console.log('Sandbox Move:', JSON.stringify(move, null, 2));

        const traceEntry = trace.entries.find((e) => e.action.moveNumber === move.moveNumber);
        if (traceEntry) {
          console.log(
            'Sandbox State Summary BEFORE move (from trace history):',
            JSON.stringify(traceEntry.boardBeforeSummary, null, 2)
          );
          console.log('Sandbox State Hash (before move):', traceEntry.stateHashBefore);
          console.log(
            'Sandbox State Summary AFTER move (from trace history):',
            JSON.stringify(traceEntry.boardAfterSummary, null, 2)
          );
          console.log('Sandbox State Hash (after move):', traceEntry.stateHashAfter);
        }

        const backendState = engine.getGameState();
        console.log(
          'Backend State Summary (before move):',
          JSON.stringify(summarizeBoard(backendState.board), null, 2)
        );
        console.log('Backend State Hash (before move):', hashGameState(backendState));
        console.log('Backend currentPlayer/currentPhase:', {
          currentPlayer: backendState.currentPlayer,
          currentPhase: backendState.currentPhase,
        });
        console.log('Backend Valid Moves:', JSON.stringify(backendMoves, null, 2));

        if (move.type === 'overtaking_capture' && move.from) {
          const backendBoard = backendState.board;

          const adapters: CaptureBoardAdapters = {
            isValidPosition: (pos: Position) => {
              return (
                pos.x >= 0 && pos.x < backendBoard.size && pos.y >= 0 && pos.y < backendBoard.size
              );
            },
            isCollapsedSpace: (pos: Position, board) => {
              const key = positionToString(pos);
              return board.collapsedSpaces.has(key);
            },
            getMarkerOwner: (pos: Position, board) => {
              const key = positionToString(pos);
              const marker = board.markers.get(key);
              return marker?.player;
            },
          };

          const segments = enumerateCaptureSegmentsFromBoard(
            'square8',
            backendBoard,
            move.from,
            move.player,
            adapters
          );

          console.log(
            'Sandbox-style capture segments from backend board:',
            JSON.stringify(segments, null, 2)
          );
        }

        break;
      }

      await engine.makeMove(move as Move);
    }
  });
});
