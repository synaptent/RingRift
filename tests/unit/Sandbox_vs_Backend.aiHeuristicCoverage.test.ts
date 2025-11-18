import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BOARD_CONFIGS,
  GameState,
  Move,
  Player,
  Position,
  positionToString
} from '../../src/shared/types/game';
import { hashGameState } from '../../src/shared/engine/core';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler
} from '../../src/client/sandbox/ClientSandboxEngine';

/**
 * Sandbox → Backend heuristic coverage tests.
 *
 * Goal: whenever the sandbox AI chooses an action via maybeRunAITurn, that
 * action should:
 *   1) Be legal according to the backend GameEngine.getValidMoves for the
 *      same state and player.
 *   2) Not be omitted in situations where backend still reports legal moves
 *      but the sandbox AI claims it has none (i.e. _lastAIMove is null and
 *      no state change occurs).
 *
 * This harness focuses on early-turn behaviour for the seeds/scenarios that
 * currently show stalls in ClientSandboxEngine.aiSimulation.test.ts. It can
 * be extended to deeper turn-by-turn replay once basic coverage is verified.
 */

describe('Sandbox vs Backend AI heuristic coverage (square8 focus)', () => {
  const boardTypes: BoardType[] = ['square8'];
  const playerCounts: number[] = [2, 3];

  // Limited runs for now; we care primarily about the known-problematic
  // seeds, but this also sanity-checks nearby seeds.
  const RUNS_PER_SCENARIO = 20;

  // We do not need deep playouts here – we only care about the first few
  // decisions where the sandbox AI may already diverge from backend
  // getValidMoves coverage.
  const MAX_STEPS_PER_RUN = 16;

  // For targeted deep analysis of a specific failing sandbox seed.
  const MAX_STEPS_DEEP_SEED = 2000;

  /** Tiny deterministic PRNG (same LCG as other AI simulation tests). */
  function makePrng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      // LCG parameters from Numerical Recipes
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }

  function createBackendEngine(boardType: BoardType, numPlayers: number): GameEngine {
    const timeControl = { initialTime: 600, increment: 0, type: 'blitz' as const };
    const boardConfig = BOARD_CONFIGS[boardType];

    const players: Player[] = Array.from({ length: numPlayers }, (_, idx) => {
      const playerNumber = idx + 1;
      return {
        id: `p${playerNumber}`,
        username: `Player${playerNumber}`,
        type: 'ai',
        playerNumber,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: boardConfig.ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0
      } as Player;
    });

    const engine = new GameEngine(
      'sandbox-vs-backend-heuristic-coverage',
      boardType,
      players,
      timeControl,
      false
    );
    const started = engine.startGame();
    if (!started) {
      throw new Error('Failed to start GameEngine for sandbox vs backend heuristic coverage test');
    }
    return engine;
  }

  function createSandboxEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'ai')
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends any>(choice: TChoice): Promise<any> {
        const anyChoice = choice as any;

        // For capture_direction choices, deterministically pick the option
        // with the smallest landing x,y to keep behaviour reproducible under
        // a fixed Math.random seeding, mirroring the sandbox AI simulation
        // tests.
        if (anyChoice.type === 'capture_direction') {
          const options = anyChoice.options || [];
          if (options.length === 0) {
            throw new Error('SandboxInteractionHandler: no options for capture_direction');
          }

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
            choiceId: anyChoice.id,
            playerNumber: anyChoice.playerNumber,
            choiceType: anyChoice.type,
            selectedOption: selected
          };
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption
        };
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function positionsEqual(a?: Position, b?: Position): boolean {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a.x === b.x && a.y === b.y && (a.z ?? 0) === (b.z ?? 0);
  }

  function describeMoveForLog(move: Move): string {
    const parts: string[] = [];
    parts.push(`type=${move.type}`);
    parts.push(`player=${move.player}`);
    if (move.from) {
      parts.push(`from=${positionToString(move.from)}`);
    }
    if (move.to) {
      parts.push(`to=${positionToString(move.to)}`);
    }
    if (move.captureTarget) {
      parts.push(`captureTarget=${positionToString(move.captureTarget)}`);
    }
    if (typeof move.placementCount === 'number') {
      parts.push(`placementCount=${move.placementCount}`);
    }
    return parts.join(',');
  }

  function describeMovesListForLog(moves: Move[]): string {
    if (!moves.length) return '(none)';
    return moves.map(describeMoveForLog).join(' | ');
  }

  function buildBackendMovementSummaryForMismatch(
    sandboxMove: Move,
    backendMoves: Move[]
  ): string {
    const movementLikeBackendMoves = backendMoves.filter(
      m => m.type === 'move_ring' || m.type === 'move_stack'
    );

    const sameFromBackendMoves = movementLikeBackendMoves.filter(m =>
      positionsEqual(m.from, sandboxMove.from)
    );

    const lines: string[] = [];
    lines.push(
      `backend movement-like moves for player ${sandboxMove.player} (total ${movementLikeBackendMoves.length}):`
    );
    lines.push(`  all movement-like moves: ${describeMovesListForLog(movementLikeBackendMoves)}`);

    const fromLabel = sandboxMove.from ? positionToString(sandboxMove.from) : 'n/a';
    lines.push(
      `  movement-like moves from sandboxMove.from=${fromLabel} (total ${sameFromBackendMoves.length}):`
    );
    lines.push(`    ${describeMovesListForLog(sameFromBackendMoves)}`);

    return lines.join('\n');
  }

  function movesLooselyMatch(a: Move, b: Move): boolean {
    if (a.player !== b.player) return false;

    // Treat simple non-capture movements as equivalent whether they are
    // labelled move_ring (sandbox legacy) or move_stack (backend canonical),
    // as long as from/to match.
    const isSimpleMovementPair =
      (a.type === 'move_ring' && b.type === 'move_stack') ||
      (a.type === 'move_stack' && b.type === 'move_ring') ||
      (a.type === 'move_ring' && b.type === 'move_ring') ||
      (a.type === 'move_stack' && b.type === 'move_stack');

    if (isSimpleMovementPair) {
      return positionsEqual(a.from, b.from) && positionsEqual(a.to, b.to);
    }

    if (a.type !== b.type) return false;

    // For placement moves, we only care that both place on the same
    // destination; placementCount and other metadata can differ.
    if (a.type === 'place_ring') {
      return positionsEqual(a.to, b.to);
    }

    // For overtaking captures, require from, captureTarget, and landing.
    if (a.type === 'overtaking_capture') {
      return (
        positionsEqual(a.from, b.from) &&
        positionsEqual(a.captureTarget, b.captureTarget) &&
        positionsEqual(a.to, b.to)
      );
    }

    // For other move types (build_stack, etc.), we are not currently
    // expecting the sandbox AI to generate them. Treat them as
    // non-matching to surface any unexpected occurrences.
    return false;
  }

  function findMatchingBackendMove(sandboxMove: Move, backendMoves: Move[]): Move | null {
    for (const candidate of backendMoves) {
      if (movesLooselyMatch(sandboxMove, candidate)) {
        return candidate;
      }
    }
    return null;
  }

  for (const boardType of boardTypes) {
    for (const numPlayers of playerCounts) {
      const scenarioLabel = `${boardType} with ${numPlayers} AI players`;

      test(
        `${scenarioLabel}: sandbox AI moves are always legal according to backend getValidMoves on early turns`,
        async () => {
          const boardIndex = boardTypes.indexOf(boardType);
          const playerCountIndex = playerCounts.indexOf(numPlayers);

          for (let run = 0; run < RUNS_PER_SCENARIO; run++) {
            const seed = 1 + run + playerCountIndex * 1000 + boardIndex * 100000;
            const rng = makePrng(seed);

            const backend = createBackendEngine(boardType, numPlayers);
            const sandbox = createSandboxEngine(boardType, numPlayers);

            const originalRandom = Math.random;
            Math.random = rng;

            try {
              for (let step = 0; step < MAX_STEPS_PER_RUN; step++) {
                // Advance backend through any automatic line/territory
                // phases so that getValidMoves reflects a
                // player-actionable phase, mirroring sandbox integration
                // of these phases into its movement flow.
                backend.stepAutomaticPhasesForTesting();

                const backendBefore = backend.getGameState();
                const sandboxBefore = sandbox.getGameState();

                // If either engine is no longer active, stop this run early.
                if (
                  backendBefore.gameStatus !== 'active' ||
                  sandboxBefore.gameStatus !== 'active'
                ) {
                  break;
                }

                // For early-turn heuristic coverage we expect the current
                // player to be aligned as long as we apply sandbox-chosen
                // moves back into the backend.
                if (backendBefore.currentPlayer !== sandboxBefore.currentPlayer) {
                  throw new Error(
                    `Pre-step desync in heuristic coverage harness: scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}, ` +
                      `backendCurrent=${backendBefore.currentPlayer}, sandboxCurrent=${sandboxBefore.currentPlayer}, ` +
                      `backendPhase=${backendBefore.currentPhase}, sandboxPhase=${sandboxBefore.currentPhase}`
                  );
                }

                const currentPlayer = sandboxBefore.currentPlayer;
                const backendMoves = backend.getValidMoves(currentPlayer);

                const sandboxBeforeHash = hashGameState(sandboxBefore);

                await sandbox.maybeRunAITurn();

                const sandboxAfter = sandbox.getGameState();
                const sandboxAfterHash = hashGameState(sandboxAfter);
                const sandboxMove = sandbox.getLastAIMoveForTesting();

                // Case 1: sandbox AI produced no logical move this tick.
                if (!sandboxMove) {
                  // If the sandbox state did not change and the game remains
                  // active while backend still reports legal moves, this is a
                  // pure heuristic under-coverage: sandbox AI failed to act
                  // where backend believes actions are possible.
                  if (
                    sandboxBeforeHash === sandboxAfterHash &&
                    sandboxAfter.gameStatus === 'active' &&
                    backendMoves.length > 0
                  ) {
                    throw new Error(
                      `Sandbox AI produced no move but backend has ${backendMoves.length} legal moves; ` +
                        `scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}, player=${currentPlayer}`
                    );
                  }

                  // Otherwise, either the sandbox state changed (e.g. via
                  // forced elimination or victory) or backend also has no
                  // legal moves. In both cases, there is nothing further to
                  // check on this step; continue to the next one.
                  continue;
                }

                // Case 2: sandbox AI produced a logical move. It must be
                // present in backend getValidMoves for the same player.
                const matchingBackendMove = findMatchingBackendMove(
                  sandboxMove,
                  backendMoves
                );

                if (!matchingBackendMove) {
                  const debugInfo = buildBackendMovementSummaryForMismatch(
                    sandboxMove,
                    backendMoves
                  );

                  throw new Error(
                    `Sandbox AI move is not legal according to backend getValidMoves; ` +
                      `scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                      `sandboxMove=${describeMoveForLog(sandboxMove)}, backendMovesCount=${backendMoves.length}` +
                      `\n${debugInfo}`
                  );
                }

                // Apply the matching backend move so that subsequent steps
                // continue from aligned states as far as possible.
                const { id, timestamp, moveNumber, ...payload } = matchingBackendMove;
                const result = await backend.makeMove(
                  payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
                );

                if (!result.success) {
                  throw new Error(
                    `Backend makeMove failed when applying sandbox AI move; ` +
                      `scenario=${scenarioLabel}, run=${run}, seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                      `sandboxMove=${describeMoveForLog(sandboxMove)}, error=${result.error}`
                  );
                }
              }
            } finally {
              Math.random = originalRandom;
            }
          }
        }
      );
    }
  }

  test(
    'square8 with 2 AI players / seed=14: sandbox AI moves remain legal and not under-covered up to 2000 steps',
    async () => {
      const boardType: BoardType = 'square8';
      const numPlayers = 2;
      const seed = 14; // Known failing sandbox AI simulation seed (square8 / 2p)
      const scenarioLabel = `${boardType} with ${numPlayers} AI players (deep seed ${seed})`;

      const rng = makePrng(seed);
      const backend = createBackendEngine(boardType, numPlayers);
      const sandbox = createSandboxEngine(boardType, numPlayers);

      const originalRandom = Math.random;
      Math.random = rng;

      try {
        for (let step = 0; step < MAX_STEPS_DEEP_SEED; step++) {
          // As in the early-turn harness, ensure the backend has
          // progressed through any automatic bookkeeping phases so
          // getValidMoves is evaluated from a player-actionable phase.
          backend.stepAutomaticPhasesForTesting();

          const backendBefore = backend.getGameState();
          const sandboxBefore = sandbox.getGameState();

          if (
            backendBefore.gameStatus !== 'active' ||
            sandboxBefore.gameStatus !== 'active'
          ) {
            break;
          }

          if (backendBefore.currentPlayer !== sandboxBefore.currentPlayer) {
            throw new Error(
              `Pre-step desync in deep-seed heuristic coverage: scenario=${scenarioLabel}, seed=${seed}, step=${step}, ` +
                `backendCurrent=${backendBefore.currentPlayer}, sandboxCurrent=${sandboxBefore.currentPlayer}, ` +
                `backendPhase=${backendBefore.currentPhase}, sandboxPhase=${sandboxBefore.currentPhase}`
            );
          }

          const currentPlayer = sandboxBefore.currentPlayer;
          const backendMoves = backend.getValidMoves(currentPlayer);
          const sandboxBeforeHash = hashGameState(sandboxBefore);

          await sandbox.maybeRunAITurn();

          const sandboxAfter = sandbox.getGameState();
          const sandboxAfterHash = hashGameState(sandboxAfter);
          const sandboxMove = sandbox.getLastAIMoveForTesting();

          if (!sandboxMove) {
            if (
              sandboxBeforeHash === sandboxAfterHash &&
              sandboxAfter.gameStatus === 'active' &&
              backendMoves.length > 0
            ) {
              throw new Error(
                `Sandbox AI produced no move but backend has ${backendMoves.length} legal moves; ` +
                  `scenario=${scenarioLabel}, seed=${seed}, step=${step}, player=${currentPlayer}`
              );
            }

            continue;
          }

          const matchingBackendMove = findMatchingBackendMove(sandboxMove, backendMoves);

          if (!matchingBackendMove) {
            const debugInfo = buildBackendMovementSummaryForMismatch(sandboxMove, backendMoves);

            throw new Error(
              `Sandbox AI move is not legal according to backend getValidMoves (deep-seed run); ` +
                `scenario=${scenarioLabel}, seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                `sandboxMove=${describeMoveForLog(sandboxMove)}, backendMovesCount=${backendMoves.length}` +
                `\n${debugInfo}`
            );
          }

          const { id, timestamp, moveNumber, ...payload } = matchingBackendMove;
          const result = await backend.makeMove(
            payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
          );

          if (!result.success) {
            throw new Error(
              `Backend makeMove failed when applying sandbox AI move (deep-seed run); ` +
                `scenario=${scenarioLabel}, seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                `sandboxMove=${describeMoveForLog(sandboxMove)}, error=${result.error}`
            );
          }
        }
      } finally {
        Math.random = originalRandom;
      }
    }
  );

  test(
    'square8 with 2 AI players / seed=5: sandbox AI moves remain legal and not under-covered up to 2000 steps',
    async () => {
      const boardType: BoardType = 'square8';
      const numPlayers = 2;
      const seed = 5; // Sandbox AI simulation seed for run=4 (square8 / 2p)
      const scenarioLabel = `${boardType} with ${numPlayers} AI players (deep seed ${seed})`;

      const rng = makePrng(seed);
      const backend = createBackendEngine(boardType, numPlayers);
      const sandbox = createSandboxEngine(boardType, numPlayers);

      const originalRandom = Math.random;
      Math.random = rng;

      try {
        for (let step = 0; step < MAX_STEPS_DEEP_SEED; step++) {
          backend.stepAutomaticPhasesForTesting();

          const backendBefore = backend.getGameState();
          const sandboxBefore = sandbox.getGameState();

          if (
            backendBefore.gameStatus !== 'active' ||
            sandboxBefore.gameStatus !== 'active'
          ) {
            break;
          }

          if (backendBefore.currentPlayer !== sandboxBefore.currentPlayer) {
            throw new Error(
              `Pre-step desync in deep-seed heuristic coverage: scenario=${scenarioLabel}, seed=${seed}, step=${step}, ` +
                `backendCurrent=${backendBefore.currentPlayer}, sandboxCurrent=${sandboxBefore.currentPlayer}, ` +
                `backendPhase=${backendBefore.currentPhase}, sandboxPhase=${sandboxBefore.currentPhase}`
            );
          }

          const currentPlayer = sandboxBefore.currentPlayer;
          const backendMoves = backend.getValidMoves(currentPlayer);
          const sandboxBeforeHash = hashGameState(sandboxBefore);

          await sandbox.maybeRunAITurn();

          const sandboxAfter = sandbox.getGameState();
          const sandboxAfterHash = hashGameState(sandboxAfter);
          const sandboxMove = sandbox.getLastAIMoveForTesting();

          if (!sandboxMove) {
            if (
              sandboxBeforeHash === sandboxAfterHash &&
              sandboxAfter.gameStatus === 'active' &&
              backendMoves.length > 0
            ) {
              throw new Error(
                `Sandbox AI produced no move but backend has ${backendMoves.length} legal moves; ` +
                  `scenario=${scenarioLabel}, seed=${seed}, step=${step}, player=${currentPlayer}`
              );
            }

            continue;
          }

          const matchingBackendMove = findMatchingBackendMove(sandboxMove, backendMoves);

          if (!matchingBackendMove) {
            const debugInfo = buildBackendMovementSummaryForMismatch(sandboxMove, backendMoves);

            throw new Error(
              `Sandbox AI move is not legal according to backend getValidMoves (deep-seed run); ` +
                `scenario=${scenarioLabel}, seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                `sandboxMove=${describeMoveForLog(sandboxMove)}, backendMovesCount=${backendMoves.length}` +
                `\n${debugInfo}`
            );
          }

          const { id, timestamp, moveNumber, ...payload } = matchingBackendMove;
          const result = await backend.makeMove(
            payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
          );

          if (!result.success) {
            throw new Error(
              `Backend makeMove failed when applying sandbox AI move (deep-seed run); ` +
                `scenario=${scenarioLabel}, seed=${seed}, step=${step}, player=${currentPlayer}, ` +
                `sandboxMove=${describeMoveForLog(sandboxMove)}, error=${result.error}`
            );
          }
        }
      } finally {
        Math.random = originalRandom;
      }
    }
  );

  // Optional diagnostic: backend movement set around the first movement-phase turn
  // for a specific seed. This is skipped by default to avoid noisy output in CI,
  // but can be enabled locally when investigating movement semantics.
  test.skip(
    'DIAGNOSTIC ONLY: backend movement moves at first movement-phase turn for square8 / 2p / seed=1',
    async () => {
      const boardType: BoardType = 'square8';
      const numPlayers = 2;
      const seed = 1;
      const rng = makePrng(seed);

      const backend = createBackendEngine(boardType, numPlayers);

      const originalRandom = Math.random;
      Math.random = rng;

      try {
        // Drive the backend AI until we encounter the first explicit movement
        // phase. At that point, log all movement-like moves for inspection.
        // We cap the loop to avoid accidental infinite runs if something
        // changes in the rules.
        const MAX_BACKEND_STEPS = 64;

        for (let step = 0; step < MAX_BACKEND_STEPS; step++) {
          const state = backend.getGameState();

          if (state.gameStatus !== 'active') {
            console.log(
              '[diagnostic] game ended before reaching a movement phase:',
              state.gameStatus,
              'phase=',
              state.currentPhase
            );
            break;
          }

          const backendMoves = backend.getValidMoves(state.currentPlayer);

          // If we are in movement phase, dump the movement-like moves and stop.
          if (state.currentPhase === 'movement') {
            const movementLike = backendMoves.filter(
              m => m.type === 'move_ring' || m.type === 'move_stack'
            );

            console.log(
              `[diagnostic] First movement-phase turn reached at step ${step}, ` +
                `player=${state.currentPlayer}, backendMovesCount=${backendMoves.length}`
            );
            console.log(
              '[diagnostic] All movement-like backend moves:',
              describeMovesListForLog(movementLike)
            );
            break;
          }

          if (!backendMoves.length) {
            console.log(
              '[diagnostic] No backend moves available at step',
              step,
              'phase=',
              state.currentPhase
            );
            break;
          }

          // Pick a random backend move using the deterministic RNG to keep
          // this diagnostic reproducible.
          const idx = Math.floor(rng() * backendMoves.length);
          const move = backendMoves[idx];
          const { id, timestamp, moveNumber, ...payload } = move;
          const result = await backend.makeMove(
            payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
          );

          if (!result.success) {
            console.log(
              '[diagnostic] backend makeMove failed at step',
              step,
              'move=',
              describeMoveForLog(move),
              'error=',
              result.error
            );
            break;
          }
        }
      } finally {
        Math.random = originalRandom;
      }
    }
  );
});
