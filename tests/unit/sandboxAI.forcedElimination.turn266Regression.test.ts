import * as fs from 'fs';
import * as path from 'path';
import type { GameState, Move } from '../../src/shared/types/game';
import {
  deserializeGameState,
  type SerializedGameState,
} from '../../src/shared/engine/contracts/serialization';
import { getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { computeGlobalLegalActionsSummary } from '../../src/shared/engine/globalActions';
import {
  maybeRunAITurnSandbox,
  resetSandboxAIStallCounters,
  type SandboxAIHooks,
} from '../../src/client/sandbox/sandboxAI';
import { hashGameState } from '../../src/shared/engine';

/**
 * Helper to load the shared turn‑266 sandbox scenario fixture and rehydrate it
 * into a canonical GameState. This uses the same SerializedGameState schema
 * as ClientSandboxEngine.initFromSerializedState.
 */
function loadTurn266SerializedState(): SerializedGameState {
  // Test files live under <repo-root>/tests/unit, while the fixture JSON is
  // stored at the project root. Walking up two levels from __dirname yields
  // the repo root in both local and CI environments.
  const fixturePath = path.join(
    __dirname,
    '../..',
    'ringrift_scenario_sandbox_scenario_turn_266.json'
  );
  const raw = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as {
    state: SerializedGameState;
  };
  return raw.state;
}

function loadTurn266GameStateForPlayer2AI(): GameState {
  const serialized = loadTurn266SerializedState();
  const baseState = deserializeGameState(serialized);

  // For sandbox AI tests we treat both players as AI seats so that
  // maybeRunAITurnSandbox will execute for the current player.
  return {
    ...baseState,
    players: baseState.players.map((p) => ({
      ...p,
      type: 'ai' as const,
    })),
  };
}

describe('turn‑266 forced_elimination sandbox scenario – engine surface', () => {
  it('exposes forced_elimination candidates in getValidMoves and global action summary', () => {
    const serialized = loadTurn266SerializedState();
    const state = deserializeGameState(serialized);

    expect(state.boardType).toBe('hexagonal');
    expect(state.currentPhase).toBe('forced_elimination');
    expect(state.currentPlayer).toBe(2);
    expect(state.gameStatus).toBe('active');

    const summary = computeGlobalLegalActionsSummary(state, state.currentPlayer);

    // These expectations document the current canonical summary for this
    // regression fixture. If they change, the turn‑266 characterization has
    // changed and this test should be updated alongside any engine fixes.
    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasGlobalPlacementAction).toBe(false);
    expect(summary.hasForcedEliminationAction).toBe(true);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(true);

    const allMoves = getValidMoves(state);
    const feMoves = allMoves.filter((m) => m.type === 'forced_elimination');

    // Characterize the engine behaviour at this exact state.
    expect(allMoves.length).toBeGreaterThan(0);
    expect(feMoves.length).toBeGreaterThan(0);
    expect(new Set(allMoves.map((m) => m.type))).toEqual(new Set(['forced_elimination']));
  });
});

describe('turn‑266 forced_elimination sandbox scenario – sandbox AI behaviour', () => {
  it('runs maybeRunAITurnSandbox once and applies a forced_elimination move for Player 2', async () => {
    let currentState = loadTurn266GameStateForPlayer2AI();

    // Sanity‑check starting conditions match the fixture.
    expect(currentState.currentPhase).toBe('forced_elimination');
    expect(currentState.currentPlayer).toBe(2);
    expect(currentState.gameStatus).toBe('active');

    const getValidMovesSpy = jest.fn(() => getValidMoves(currentState));

    const applyCanonicalMove = jest.fn(async (move: Move) => {
      // Sandbox-only FE application stub for this regression test:
      // - Assert that maybeRunAITurnSandbox selected a forced_elimination move.
      // - Apply a minimal, deterministic state change so that:
      //   - hashGameState(before) !== hashGameState(after), and
      //   - either currentPhase or currentPlayer (or both) differ from the
      //     original FE state, allowing the stall detector to observe progress.
      expect(move.type).toBe('forced_elimination');
      const playerIndex = currentState.players.findIndex((p) => p.playerNumber === move.player);
      const elimCount =
        (Array.isArray((move as any).eliminatedRings) && (move as any).eliminatedRings[0]?.count) ||
        1;

      const players = [...currentState.players];
      if (playerIndex >= 0) {
        players[playerIndex] = {
          ...players[playerIndex],
          eliminatedRings: players[playerIndex].eliminatedRings + elimCount,
        };
      }

      currentState = {
        ...currentState,
        players,
        totalRingsEliminated: currentState.totalRingsEliminated + elimCount,
        // Advance to the next player's ring_placement phase to mirror the
        // canonical "end-of-turn after elimination" shape at a high level.
        currentPlayer: currentState.currentPlayer === 1 ? 2 : 1,
        currentPhase: 'ring_placement' as GameState['currentPhase'],
      };
    });

    const hooks: SandboxAIHooks = {
      getPlayerStacks: () => [],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      getValidMovesForCurrentPlayer: () => getValidMovesSpy(),
      createHypotheticalBoardWithPlacement: (board) => board,
      tryPlaceRings: async () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {},
      appendHistoryEntry: () => {},
      getGameState: () => currentState,
      setGameState: (state: GameState) => {
        currentState = state;
      },
      setLastAIMove: () => {},
      setSelectedStackKey: () => {},
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove,
      hasPendingTerritorySelfElimination: () => false,
      hasPendingLineRewardElimination: () => false,
      canCurrentPlayerSwapSides: () => false,
      applySwapSidesForCurrentPlayer: () => false,
    };

    resetSandboxAIStallCounters();

    const beforeHash = hashGameState(currentState);

    await maybeRunAITurnSandbox(hooks, () => 0.5);

    // Confirm that the FE branch was exercised with concrete candidates.
    expect(getValidMovesSpy).toHaveBeenCalledTimes(1);
    const allMoves = getValidMovesSpy.mock.results[0].value as Move[];
    const feCandidates = allMoves.filter((m) => m.type === 'forced_elimination');
    expect(feCandidates.length).toBeGreaterThan(0);

    // Confirm that exactly one canonical move was applied and that it is a
    // forced_elimination move for Player 2.
    expect(applyCanonicalMove).toHaveBeenCalledTimes(1);
    const applied = applyCanonicalMove.mock.calls[0][0] as Move;
    expect(applied.type).toBe('forced_elimination');
    expect(applied.player).toBe(2);

    // The sandbox AI turn must be observable as progress by the stall detector:
    // the game state hash must change and either the phase or current player
    // (or both) must advance away from the original FE state.
    const afterHash = hashGameState(currentState);
    expect(afterHash).not.toBe(beforeHash);
    expect(
      currentState.currentPhase !== 'forced_elimination' || currentState.currentPlayer !== 2
    ).toBe(true);
  });
});
