import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BOARD_CONFIGS,
  type BoardType,
  type GameState,
  type Move,
  type Player,
  type TimeControl,
} from '../../src/shared/types/game';
import { computeProgressSnapshot, summarizeBoard } from '../../src/shared/engine/core';
import { SeededRNG } from '../../src/shared/utils/rng';

/**
 * Targeted reproduction of the orchestrator S-invariant regression surfaced
 * by scripts/run-orchestrator-soak.ts.
 *
 * This test mirrors the backend host wiring used by the soak harness:
 * - GameEngine with orchestrator adapter enabled.
 * - Deterministic SeededRNG-driven move selection.
 *
 * Once the underlying bug is fixed, S should remain non-decreasing for this
 * seeded game just as in the mutator-level invariant tests.
 */

const boardType: BoardType = 'square8';

const timeControl: TimeControl = {
  initialTime: 600,
  increment: 0,
  type: 'blitz',
};

function createDefaultPlayers(): Player[] {
  const ringsPerPlayer = BOARD_CONFIGS[boardType].ringsPerPlayer;
  return [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

function createBackendOrchestratorHost(gameId: string, gameSeed: number): GameEngine {
  const players = createDefaultPlayers();

  const engine = new GameEngine(
    gameId,
    boardType,
    players,
    timeControl,
    false,
    undefined,
    gameSeed
  );

  // Ensure orchestrator adapter is enabled regardless of config flags.
  engine.enableOrchestratorAdapter();

  // Mark all players ready and seed rng for reproducibility, mirroring
  // scripts/run-orchestrator-soak.ts.
  const engineAny = engine as any;
  if (engineAny.gameState && Array.isArray(engineAny.gameState.players)) {
    engineAny.gameState.players.forEach((p: any) => {
      p.isReady = true;
    });
    engineAny.gameState.rngSeed = gameSeed;
  }

  const started = engine.startGame();
  if (!started) {
    throw new Error(`Failed to start GameEngine for gameId=${gameId}`);
  }

  return engine;
}

function toEngineMove(move: Move): Omit<Move, 'id' | 'timestamp' | 'moveNumber'> {
  const { id, timestamp, moveNumber, ...rest } = move as any;
  return {
    ...rest,
    thinkTime: 0,
  };
}

// Seeds captured from results/orchestrator_soak_smoke.json where the
// orchestrator soak harness reported S_INVARIANT_DECREASED for backend
// games on square8. These are kept in a single place so new regressions
// can be promoted from soak output into this targeted harness.
const REGRESSION_SEEDS: number[] = [786238345, 265064459];

/**
 * Backend GameEngine S-invariant regression tests.
 *
 * These tests verify that the progress invariant S (markers + collapsed + eliminated)
 * is non-decreasing throughout a game when using the backend GameEngine with FSM
 * orchestration.
 *
 * The backend GameEngine now uses FSM for move validation and orchestration via:
 * - TurnEngineAdapter -> turnOrchestrator.processTurnAsync
 * - turnOrchestrator uses validateMoveWithFSM and computeFSMOrchestration
 */
describe('Orchestrator S-invariant – backend harness parity', () => {
  REGRESSION_SEEDS.forEach((seed) => {
    it(`S remains non-decreasing for a short seeded backend orchestrator game (square8, seed=${seed})`, async () => {
      const engine = createBackendOrchestratorHost(`s-invariant-orchestrator-seed-${seed}`, seed);
      const moveRng = new SeededRNG(seed);

      const maxTurns = 20;
      let lastS: number | null = null;
      let lastAppliedMove: Move | null = null;
      let lastState: GameState | null = null;

      for (let turn = 0; turn < maxTurns; turn += 1) {
        const state: GameState = engine.getGameState() as GameState;
        const snapshot = computeProgressSnapshot(state as any);

        if (lastS !== null) {
          if (snapshot.S < lastS) {
            // Emit a focused debug payload so we can inspect the exact
            // move and S components responsible for the regression.
            // eslint-disable-next-line no-console
            console.log('Orchestrator S-invariant decreased', {
              seed,
              turn,
              previousS: lastS,
              currentS: snapshot.S,
              markers: snapshot.markers,
              collapsed: snapshot.collapsed,
              eliminated: snapshot.eliminated,
              lastAppliedMove,
              previousBoardSummary: lastState ? summarizeBoard(lastState.board) : null,
              currentBoardSummary: summarizeBoard(state.board),
            });
          }

          // Progress invariant: S must be non-decreasing across canonical turns.
          expect(snapshot.S).toBeGreaterThanOrEqual(lastS);
        }
        lastS = snapshot.S;
        lastState = state;

        if (state.gameStatus !== 'active') {
          break;
        }

        const currentPlayer = state.currentPlayer;
        const moves = (engine as any as GameEngine).getValidMoves(currentPlayer) as Move[];
        expect(moves.length).toBeGreaterThan(0);

        const realMoveTypes: Move['type'][] = [
          'place_ring',
          'skip_placement',
          'move_stack',
          'move_stack',
          'overtaking_capture',
          'continue_capture_segment',
        ];
        const realMoves = moves.filter((m) => realMoveTypes.includes(m.type));
        const candidateMoves = realMoves.length > 0 ? realMoves : moves;

        const selectedIndex = moveRng.nextInt(0, candidateMoves.length);
        const selectedMove = candidateMoves[selectedIndex];
        lastAppliedMove = selectedMove;

        const result = await engine.makeMove(toEngineMove(selectedMove));
        expect(result.success).toBe(true);
      }
    });
  });
});
