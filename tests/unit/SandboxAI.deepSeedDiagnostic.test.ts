/**
 * Diagnostic test to investigate deep-seed AI stalls in sandbox vs backend.
 *
 * This test replicates the exact conditions at the stall point and provides
 * detailed logging to understand why sandbox AI has no moves while backend does.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BOARD_CONFIGS,
  GameState,
  Move,
  Player,
  positionToString,
} from '../../src/shared/types/game';
import { hashGameState } from '../../src/shared/engine/core';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { enumerateSimpleMovesForPlayer } from '../../src/shared/engine/aggregates/MovementAggregate';
import { enumerateAllCaptureMoves } from '../../src/shared/engine/aggregates/CaptureAggregate';
import { getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
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
      territorySpaces: 0,
    } as Player;
  });

  const engine = new GameEngine('deep-seed-diagnostic', boardType, players, timeControl, false);
  engine.startGame();
  return engine;
}

function createSandboxEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers,
    playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
  };

  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice>(choice: TChoice): Promise<any> {
      const anyChoice = choice as any;
      if (anyChoice.type === 'capture_direction') {
        const options = anyChoice.options || [];
        if (options.length === 0) {
          throw new Error('No options for capture_direction');
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
          selectedOption: selected,
        };
      }
      const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption,
      };
    },
  };

  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

function findMatchingBackendMove(sandboxMove: Move, backendMoves: Move[]): Move | null {
  for (const candidate of backendMoves) {
    if (candidate.player !== sandboxMove.player) continue;

    // Simple movement match
    if (
      (sandboxMove.type === 'move_ring' || sandboxMove.type === 'move_stack') &&
      (candidate.type === 'move_ring' || candidate.type === 'move_stack')
    ) {
      if (
        sandboxMove.from?.x === candidate.from?.x &&
        sandboxMove.from?.y === candidate.from?.y &&
        sandboxMove.to?.x === candidate.to?.x &&
        sandboxMove.to?.y === candidate.to?.y
      ) {
        return candidate;
      }
    }

    // Placement match - must also match placementCount
    if (sandboxMove.type === 'place_ring' && candidate.type === 'place_ring') {
      const sandboxCount = (sandboxMove as any).placementCount ?? 1;
      const candidateCount = (candidate as any).placementCount ?? 1;
      if (
        sandboxMove.to?.x === candidate.to?.x &&
        sandboxMove.to?.y === candidate.to?.y &&
        sandboxCount === candidateCount
      ) {
        return candidate;
      }
    }

    // Capture match
    if (sandboxMove.type === 'overtaking_capture' && candidate.type === 'overtaking_capture') {
      if (
        sandboxMove.from?.x === candidate.from?.x &&
        sandboxMove.from?.y === candidate.from?.y &&
        sandboxMove.to?.x === candidate.to?.x &&
        sandboxMove.to?.y === candidate.to?.y &&
        sandboxMove.captureTarget?.x === candidate.captureTarget?.x &&
        sandboxMove.captureTarget?.y === candidate.captureTarget?.y
      ) {
        return candidate;
      }
    }

    // Skip placement
    if (sandboxMove.type === 'skip_placement' && candidate.type === 'skip_placement') {
      return candidate;
    }
  }
  return null;
}

/**
 * Check if backend and sandbox states have any stack mismatches.
 * Returns a description of the first mismatch found, or null if states match.
 */
function checkStackMismatch(backendState: GameState, sandboxState: GameState): string | null {
  const backendStacks = backendState.board.stacks;
  const sandboxStacks = sandboxState.board.stacks;

  // Check for stacks only in backend
  for (const [key, bStack] of backendStacks) {
    const sStack = sandboxStacks.get(key);
    if (!sStack) {
      return `Stack at ${key} exists in backend (h=${bStack.stackHeight}) but not in sandbox`;
    }
    if (bStack.stackHeight !== sStack.stackHeight) {
      return `Stack at ${key} height mismatch: backend=${bStack.stackHeight}, sandbox=${sStack.stackHeight}`;
    }
    if (bStack.controllingPlayer !== sStack.controllingPlayer) {
      return `Stack at ${key} owner mismatch: backend=${bStack.controllingPlayer}, sandbox=${sStack.controllingPlayer}`;
    }
    if (bStack.capHeight !== sStack.capHeight) {
      return `Stack at ${key} cap mismatch: backend=${bStack.capHeight}, sandbox=${sStack.capHeight}`;
    }
  }

  // Check for stacks only in sandbox
  for (const [key, sStack] of sandboxStacks) {
    if (!backendStacks.has(key)) {
      return `Stack at ${key} exists in sandbox (h=${sStack.stackHeight}) but not in backend`;
    }
  }

  // Check player rings in hand
  for (const sPlayer of sandboxState.players) {
    const bPlayer = backendState.players.find((p) => p.playerNumber === sPlayer.playerNumber);
    if (bPlayer && bPlayer.ringsInHand !== sPlayer.ringsInHand) {
      return `Player ${sPlayer.playerNumber} ringsInHand mismatch: backend=${bPlayer.ringsInHand}, sandbox=${sPlayer.ringsInHand}`;
    }
  }

  return null;
}

async function resolveBackendChainIfPresent(backend: GameEngine): Promise<void> {
  const MAX_STEPS = 32;
  let steps = 0;

  for (;;) {
    const state = backend.getGameState();
    if (state.currentPhase !== 'chain_capture' || state.gameStatus !== 'active') {
      break;
    }

    steps++;
    if (steps > MAX_STEPS) {
      throw new Error('resolveBackendChainIfPresent: exceeded maximum chain-capture steps');
    }

    const moves = backend.getValidMoves(state.currentPlayer);
    const chainMoves = moves.filter((m) => m.type === 'continue_capture_segment');

    if (chainMoves.length === 0) {
      break;
    }

    const next = chainMoves.reduce((best, current) => {
      if (!best.to || !current.to) return best;
      if (current.to.x < best.to.x) return current;
      if (current.to.x > best.to.x) return best;
      if (current.to.y < best.to.y) return current;
      return best;
    }, chainMoves[0]);

    const { id, timestamp, moveNumber, ...payload } = next as any;
    await backend.makeMove(payload);
  }
}

describe('Deep seed diagnostic for sandbox AI stalls', () => {
  // Test both problematic seeds
  test.each([
    { seed: 5, expectedStallStep: 44 },
    { seed: 14, expectedStallStep: 42 },
  ])(
    'Seed $seed diagnostic: investigate stall at approximately step $expectedStallStep',
    async ({ seed }) => {
      const boardType: BoardType = 'square8';
      const numPlayers = 2;
      const targetStep = 2000; // Run to completion or actual stall

      const rng = makePrng(seed);

      // IMPORTANT: Replace Math.random BEFORE creating engines to ensure
      // generateGameSeed() inside ClientSandboxEngine uses the seeded RNG.
      // This prevents flakiness when other tests consume Math.random first.
      const originalRandom = Math.random;
      Math.random = rng;

      const backend = createBackendEngine(boardType, numPlayers);
      const sandbox = createSandboxEngine(boardType, numPlayers);

      try {
        let firstDivergenceStep: number | null = null;

        // Log initial states before any moves
        console.log(`\n--- INITIAL STATE (before step 0) ---`);
        const initBackend = backend.getGameState();
        const initSandbox = sandbox.getGameState();
        console.log(
          `Backend: phase=${initBackend.currentPhase}, player=${initBackend.currentPlayer}, stacks=${initBackend.board.stacks.size}`
        );
        console.log(
          `Sandbox: phase=${initSandbox.currentPhase}, player=${initSandbox.currentPlayer}, stacks=${initSandbox.board.stacks.size}`
        );
        console.log(`Backend P1 rings: ${initBackend.players[0].ringsInHand}`);
        console.log(`Sandbox P1 rings: ${initSandbox.players[0].ringsInHand}`);

        for (let step = 0; step < targetStep; step++) {
          backend.stepAutomaticPhasesForTesting();
          await resolveBackendChainIfPresent(backend);

          const backendBefore = backend.getGameState();
          const sandboxBefore = sandbox.getGameState();

          if (backendBefore.gameStatus !== 'active' || sandboxBefore.gameStatus !== 'active') {
            console.log(
              `[Step ${step}] Game ended - backend: ${backendBefore.gameStatus}, sandbox: ${sandboxBefore.gameStatus}`
            );
            break;
          }

          // Check for structural alignment
          if (backendBefore.currentPlayer !== sandboxBefore.currentPlayer) {
            console.log(
              `[Step ${step}] Player desync - backend: ${backendBefore.currentPlayer}, sandbox: ${sandboxBefore.currentPlayer}`
            );
            break;
          }

          // Check for early state divergence BEFORE applying moves
          // This will help identify when states first diverge
          if (firstDivergenceStep === null) {
            const stackMismatch = checkStackMismatch(backendBefore, sandboxBefore);
            if (stackMismatch) {
              firstDivergenceStep = step;
              console.log(`\n========================================`);
              console.log(`[FIRST STATE DIVERGENCE at Step ${step}]`);
              console.log(`========================================`);
              console.log(
                `Phase: ${backendBefore.currentPhase}, Player: ${backendBefore.currentPlayer}`
              );
              console.log(`Mismatch: ${stackMismatch}`);

              // Show all stack differences
              console.log(`\n--- All Stack Differences ---`);
              const allStackKeys = new Set<string>([
                ...Array.from(backendBefore.board.stacks.keys() as Iterable<string>),
                ...Array.from(sandboxBefore.board.stacks.keys() as Iterable<string>),
              ]);
              allStackKeys.forEach((key: string) => {
                const bStack = backendBefore.board.stacks.get(key);
                const sStack = sandboxBefore.board.stacks.get(key);
                if (!bStack && sStack) {
                  console.log(
                    `  ${key}: ONLY IN SANDBOX - h=${sStack.stackHeight}, cap=${sStack.capHeight}, owner=${sStack.controllingPlayer}`
                  );
                } else if (bStack && !sStack) {
                  console.log(
                    `  ${key}: ONLY IN BACKEND - h=${bStack.stackHeight}, cap=${bStack.capHeight}, owner=${bStack.controllingPlayer}`
                  );
                } else if (bStack && sStack) {
                  if (
                    bStack.stackHeight !== sStack.stackHeight ||
                    bStack.controllingPlayer !== sStack.controllingPlayer ||
                    bStack.capHeight !== sStack.capHeight
                  ) {
                    console.log(
                      `  ${key}: DIFFERENT - backend(h=${bStack.stackHeight},cap=${bStack.capHeight},p=${bStack.controllingPlayer}) vs sandbox(h=${sStack.stackHeight},cap=${sStack.capHeight},p=${sStack.controllingPlayer})`
                    );
                  }
                }
              });

              // Show player rings
              console.log(`\n--- Player Rings ---`);
              sandboxBefore.players.forEach((p) => {
                const bPlayer = backendBefore.players.find(
                  (bp: Player) => bp.playerNumber === p.playerNumber
                );
                const match = p.ringsInHand === bPlayer?.ringsInHand ? '✓' : '✗';
                console.log(
                  `  P${p.playerNumber}: sandbox=${p.ringsInHand}, backend=${bPlayer?.ringsInHand} ${match}`
                );
              });

              throw new Error(
                `State divergence first detected at step ${step}. See console for details.`
              );
            }
          }

          const currentPlayer = sandboxBefore.currentPlayer;
          const backendMoves = backend.getValidMoves(currentPlayer);
          const sandboxBeforeHash = hashGameState(sandboxBefore);

          // Get sandbox's internal move enumeration for comparison
          const sandboxValidMoves = sandbox.getValidMoves(currentPlayer);

          await sandbox.maybeRunAITurn();

          const sandboxAfter = sandbox.getGameState();
          const sandboxAfterHash = hashGameState(sandboxAfter);
          const sandboxMove = sandbox.getLastAIMoveForTesting();

          // Log periodic progress and detailed step 0 info
          if (step % 10 === 0 || step === 0) {
            console.log(
              `[Step ${step}] Phase: ${sandboxBefore.currentPhase}, Player: ${currentPlayer}, ` +
                `Backend moves: ${backendMoves.length}, Sandbox internal moves: ${sandboxValidMoves.length}`
            );
          }

          // Extra logging to trace all moves
          if (step < 5) {
            console.log(`\n[Step ${step}] ----`);
            console.log(
              `  BEFORE: Backend stacks=${backendBefore.board.stacks.size}, Sandbox stacks=${sandboxBefore.board.stacks.size}`
            );
            console.log(
              `  BEFORE: Phase=${backendBefore.currentPhase}, Player=${backendBefore.currentPlayer}`
            );
            if (sandboxMove) {
              console.log(
                `  SANDBOX MOVE: type=${sandboxMove.type}, player=${sandboxMove.player}, to=${sandboxMove.to ? positionToString(sandboxMove.to) : 'n/a'}`
              );
            } else {
              console.log(`  SANDBOX MOVE: (none)`);
            }
          }

          // Detect the stall condition
          if (!sandboxMove) {
            if (
              sandboxBeforeHash === sandboxAfterHash &&
              sandboxAfter.gameStatus === 'active' &&
              backendMoves.length > 0
            ) {
              console.log('\n========================================');
              console.log(`[STALL DETECTED] Step ${step}, Player ${currentPlayer}`);
              console.log('========================================');

              console.log('\n--- State Summary ---');
              console.log(`Phase: ${sandboxBefore.currentPhase}`);
              console.log(`Backend moves count: ${backendMoves.length}`);
              console.log(`Sandbox internal moves count: ${sandboxValidMoves.length}`);

              console.log('\n--- Player Stacks ---');
              const playerStacks = Array.from(sandboxBefore.board.stacks.entries()).filter(
                ([_, s]) => s.controllingPlayer === currentPlayer
              );
              console.log(`Player ${currentPlayer} stacks: ${playerStacks.length}`);
              playerStacks.forEach(([key, stack]) => {
                console.log(
                  `  ${key}: height=${stack.stackHeight}, cap=${stack.capHeight}, rings=${JSON.stringify(stack.rings)}`
                );
              });

              console.log('\n--- Backend Moves ---');
              const movesByType = new Map<string, Move[]>();
              backendMoves.forEach((m) => {
                const list = movesByType.get(m.type) || [];
                list.push(m);
                movesByType.set(m.type, list);
              });
              movesByType.forEach((moves, type) => {
                console.log(`${type}: ${moves.length} moves`);
                moves.slice(0, 5).forEach((m) => {
                  console.log(
                    `  from=${m.from ? positionToString(m.from) : 'n/a'}, to=${m.to ? positionToString(m.to) : 'n/a'}`
                  );
                });
                if (moves.length > 5) console.log(`  ... and ${moves.length - 5} more`);
              });

              console.log('\n--- Sandbox Internal Valid Moves ---');
              console.log(`Count: ${sandboxValidMoves.length}`);
              if (sandboxValidMoves.length > 0) {
                sandboxValidMoves.slice(0, 10).forEach((m) => {
                  console.log(
                    `  ${m.type}: from=${m.from ? positionToString(m.from) : 'n/a'}, to=${m.to ? positionToString(m.to) : 'n/a'}, placementCount=${(m as any).placementCount ?? 'n/a'}`
                  );
                });
              } else {
                console.log('  (no sandbox internal moves)');
              }

              // Direct enumeration test
              console.log('\n--- Direct Enumeration (shared aggregates on sandbox state) ---');
              const directMovements = enumerateSimpleMovesForPlayer(sandboxBefore, currentPlayer);
              const directCaptures = enumerateAllCaptureMoves(sandboxBefore, currentPlayer);
              console.log(`Direct simple moves: ${directMovements.length}`);
              console.log(`Direct captures: ${directCaptures.length}`);

              // Test getValidMoves directly on sandbox state
              console.log('\n--- getValidMoves(sandboxState) ---');
              const orchestratorMoves = getValidMoves(sandboxBefore);
              console.log(`Orchestrator getValidMoves count: ${orchestratorMoves.length}`);

              // Compare sandbox and backend board states
              console.log('\n--- Board State Comparison ---');
              console.log(`Backend stacks: ${backendBefore.board.stacks.size}`);
              console.log(`Sandbox stacks: ${sandboxBefore.board.stacks.size}`);
              console.log(`Backend markers: ${backendBefore.board.markers.size}`);
              console.log(`Sandbox markers: ${sandboxBefore.board.markers.size}`);
              console.log(`Backend collapsed: ${backendBefore.board.collapsedSpaces.size}`);
              console.log(`Sandbox collapsed: ${sandboxBefore.board.collapsedSpaces.size}`);

              // Check if backend and sandbox agree on stacks at movement-relevant positions
              console.log('\n--- Stack Position Comparison ---');
              const allStackKeys = new Set<string>([
                ...Array.from(backendBefore.board.stacks.keys() as Iterable<string>),
                ...Array.from(sandboxBefore.board.stacks.keys() as Iterable<string>),
              ]);
              let mismatchCount = 0;
              allStackKeys.forEach((key: string) => {
                const bStack = backendBefore.board.stacks.get(key);
                const sStack = sandboxBefore.board.stacks.get(key);
                if (!bStack && sStack) {
                  console.log(
                    `  ${key}: ONLY IN SANDBOX - height=${sStack.stackHeight}, owner=${sStack.controllingPlayer}`
                  );
                  mismatchCount++;
                } else if (bStack && !sStack) {
                  console.log(
                    `  ${key}: ONLY IN BACKEND - height=${bStack.stackHeight}, owner=${bStack.controllingPlayer}`
                  );
                  mismatchCount++;
                } else if (bStack && sStack) {
                  if (
                    bStack.stackHeight !== sStack.stackHeight ||
                    bStack.controllingPlayer !== sStack.controllingPlayer
                  ) {
                    console.log(
                      `  ${key}: MISMATCH - backend(h=${bStack.stackHeight},p=${bStack.controllingPlayer}) vs sandbox(h=${sStack.stackHeight},p=${sStack.controllingPlayer})`
                    );
                    mismatchCount++;
                  }
                }
              });
              if (mismatchCount === 0) {
                console.log('  All stacks match between backend and sandbox');
              }

              // Check players ringsInHand
              console.log('\n--- Player Rings In Hand ---');
              sandboxBefore.players.forEach((p) => {
                const bPlayer = backendBefore.players.find(
                  (bp: Player) => bp.playerNumber === p.playerNumber
                );
                console.log(
                  `  P${p.playerNumber}: sandbox=${p.ringsInHand}, backend=${bPlayer?.ringsInHand}`
                );
              });

              // Fail the test with summary
              throw new Error(
                `Stall at step ${step}: sandbox has 0 moves, backend has ${backendMoves.length}. ` +
                  `See console output above for detailed diagnostics.`
              );
            }
            continue;
          }

          // Normal case: apply sandbox move to backend
          const matchingBackendMove = findMatchingBackendMove(sandboxMove, backendMoves);
          if (!matchingBackendMove) {
            console.log(`\n========================================`);
            console.log(`[MOVE MISMATCH at Step ${step}]`);
            console.log(`========================================`);
            console.log(`Sandbox move: ${sandboxMove.type}`);
            console.log(`  from: ${sandboxMove.from ? positionToString(sandboxMove.from) : 'n/a'}`);
            console.log(`  to: ${sandboxMove.to ? positionToString(sandboxMove.to) : 'n/a'}`);
            console.log(
              `  captureTarget: ${sandboxMove.captureTarget ? positionToString(sandboxMove.captureTarget) : 'n/a'}`
            );

            console.log(`\nBackend moves of same type (${sandboxMove.type}):`);
            const sametype = backendMoves.filter((m) => m.type === sandboxMove.type);
            if (sametype.length === 0) {
              console.log(`  (none - backend has no ${sandboxMove.type} moves)`);
            } else {
              sametype.forEach((m) => {
                console.log(
                  `  from=${m.from ? positionToString(m.from) : 'n/a'}, to=${m.to ? positionToString(m.to) : 'n/a'}, captureTarget=${m.captureTarget ? positionToString(m.captureTarget) : 'n/a'}`
                );
              });
            }

            // Check board state at mismatch point
            console.log(`\n--- Board State at Mismatch ---`);
            console.log(`Backend stacks: ${backendBefore.board.stacks.size}`);
            console.log(`Sandbox stacks: ${sandboxBefore.board.stacks.size}`);
            console.log(`Backend markers: ${backendBefore.board.markers.size}`);
            console.log(`Sandbox markers: ${sandboxBefore.board.markers.size}`);
            console.log(`Backend collapsed: ${backendBefore.board.collapsedSpaces.size}`);
            console.log(`Sandbox collapsed: ${sandboxBefore.board.collapsedSpaces.size}`);

            // Check the specific stack the sandbox is trying to move from
            if (sandboxMove.from) {
              const key = positionToString(sandboxMove.from);
              const sandboxStack = sandboxBefore.board.stacks.get(key);
              const backendStack = backendBefore.board.stacks.get(key);
              console.log(`\n--- Stack at sandbox move origin (${key}) ---`);
              console.log(
                `Sandbox: ${sandboxStack ? `height=${sandboxStack.stackHeight}, cap=${sandboxStack.capHeight}, owner=${sandboxStack.controllingPlayer}, rings=${JSON.stringify(sandboxStack.rings)}` : 'NOT FOUND'}`
              );
              console.log(
                `Backend: ${backendStack ? `height=${backendStack.stackHeight}, cap=${backendStack.capHeight}, owner=${backendStack.controllingPlayer}, rings=${JSON.stringify(backendStack.rings)}` : 'NOT FOUND'}`
              );
            }

            // Check capture target
            if (sandboxMove.captureTarget) {
              const key = positionToString(sandboxMove.captureTarget);
              const sandboxStack = sandboxBefore.board.stacks.get(key);
              const backendStack = backendBefore.board.stacks.get(key);
              console.log(`\n--- Stack at capture target (${key}) ---`);
              console.log(
                `Sandbox: ${sandboxStack ? `height=${sandboxStack.stackHeight}, cap=${sandboxStack.capHeight}, owner=${sandboxStack.controllingPlayer}` : 'NOT FOUND'}`
              );
              console.log(
                `Backend: ${backendStack ? `height=${backendStack.stackHeight}, cap=${backendStack.capHeight}, owner=${backendStack.controllingPlayer}` : 'NOT FOUND'}`
              );
            }

            // Check landing
            if (sandboxMove.to) {
              const key = positionToString(sandboxMove.to);
              const sandboxStack = sandboxBefore.board.stacks.get(key);
              const backendStack = backendBefore.board.stacks.get(key);
              const sandboxCollapsed = sandboxBefore.board.collapsedSpaces.has(key);
              const backendCollapsed = backendBefore.board.collapsedSpaces.has(key);
              console.log(`\n--- Landing position (${key}) ---`);
              console.log(
                `Sandbox stack: ${sandboxStack ? `height=${sandboxStack.stackHeight}` : 'empty'}, collapsed=${sandboxCollapsed}`
              );
              console.log(
                `Backend stack: ${backendStack ? `height=${backendStack.stackHeight}` : 'empty'}, collapsed=${backendCollapsed}`
              );
            }

            // Players state
            console.log(`\n--- Players ---`);
            sandboxBefore.players.forEach((p) => {
              const bPlayer = backendBefore.players.find(
                (bp: Player) => bp.playerNumber === p.playerNumber
              );
              console.log(
                `P${p.playerNumber}: sandbox ringsInHand=${p.ringsInHand}, backend=${bPlayer?.ringsInHand}`
              );
            });

            throw new Error(
              `Move mismatch at step ${step}. Sandbox made ${sandboxMove.type} that backend doesn't recognize.`
            );
          }

          const { id, timestamp, moveNumber, ...payload } = matchingBackendMove as any;
          const result = await backend.makeMove(payload);
          if (!result.success) {
            console.log(`[Step ${step}] Backend move failed: ${result.error}`);
            break;
          }
        }
      } finally {
        Math.random = originalRandom;
      }
    }
  );
});
