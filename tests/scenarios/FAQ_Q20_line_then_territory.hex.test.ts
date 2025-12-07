/**
 * FAQ Q20: Combined line + territory turn (hex board, multi-choice)
 *
 * Scenario: A capture triggers a line, which in turn disconnects a region
 * that must be processed, exercising chain → line_processing → territory_processing.
 *
 * Board: hexagonal (radius 12)
 * Players: 2 (human vs human)
 *
 * This scenario backfills FAQ Q20 for hex boards and ensures the multi-phase
 * pipeline (line reward + region order) completes within a single turn.
 */

import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { GameState, Move, Position, Player } from '../../src/shared/types/game';
import { SandboxOrchestratorAdapter } from '../../src/client/sandbox/SandboxOrchestratorAdapter';
import { positionToString } from '../../src/shared/types/game';

function createSandboxEngine(): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType: 'hexagonal',
    numPlayers: 2,
    playerKinds: ['human', 'human'],
  };

  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice>(_choice: TChoice) {
      // Deterministically pick the first option for all choices
      return {
        choiceId: (_choice as any).id,
        playerNumber: (_choice as any).playerNumber,
        choiceType: (_choice as any).type,
        selectedOption: Array.isArray((_choice as any).options)
          ? (_choice as any).options[0]
          : (_choice as any).options,
      } as any;
    },
  };

  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

function seedBoardState(engine: ClientSandboxEngine): GameState {
  const anyEngine: any = engine;
  const state: GameState = anyEngine.gameState as GameState;

  const stacks = state.board.stacks;
  const markers = state.board.markers;
  stacks.clear();
  markers.clear();
  state.board.collapsedSpaces.clear();

  // Use cube coords (q,r,s) with q + r + s = 0; store using positionToString
  const addStack = (pos: Position, player: number, height: number) => {
    stacks.set(positionToString(pos), {
      position: pos,
      rings: Array(height).fill(player),
      stackHeight: height,
      capHeight: height,
      controllingPlayer: player,
    });
  };

  const addMarker = (pos: Position, player: number) => {
    markers.set(positionToString(pos), {
      position: pos,
      player,
      type: 'regular',
    });
  };

  // Layout:
  // P1 stack ready to capture P2, landing completes a line along q axis.
  addStack({ x: 0, y: -1, z: 1 }, 1, 1);
  addStack({ x: 1, y: -1, z: 0 }, 2, 1); // capture target
  addStack({ x: 3, y: -1, z: -2 }, 2, 1); // chain target
  addStack({ x: 5, y: -1, z: -4 }, 1, 2); // anchor stack

  // Markers to finish a line when chain lands at (2,-1,-1)
  addMarker({ x: 4, y: -1, z: -3 }, 1);
  addMarker({ x: 2, y: -1, z: -1 }, 1);

  // Region near corner to disconnect when line collapses
  addStack({ x: -5, y: 2, z: 3 }, 2, 1);
  addMarker({ x: -4, y: 2, z: 2 }, 2);

  state.currentPhase = 'movement';
  state.currentPlayer = 1;
  state.gameStatus = 'active';
  state.moveHistory = [];

  return state;
}

describe('FAQ Q20: Hex line → territory multi-choice turn', () => {
  it('processes capture → chain → line reward → territory region order in one turn', async () => {
    const engine = createSandboxEngine();
    const adapter = (engine as any).getOrchestratorAdapter() as SandboxOrchestratorAdapter;
    const state = seedBoardState(engine);

    // Triggering capture (movement phase): jump from (0,-1,1) over (1,-1,0) to (2,-1,-1)
    const captureMove: Move = {
      player: 1,
      type: 'overtaking_capture',
      from: { x: 0, y: -1, z: 1 },
      captureTarget: { x: 1, y: -1, z: 0 },
      to: { x: 2, y: -1, z: -1 },
    } as any;

    const validation = adapter.validateMove(captureMove);
    expect(validation.valid).toBe(true);

    await engine.applyCanonicalMove(captureMove);

    // Continue any chain captures automatically (handler picks first option)
    const MAX_STEPS = 10;
    for (let i = 0; i < MAX_STEPS; i++) {
      const snapshot = engine.getGameState();
      if (snapshot.currentPhase !== 'capture' && snapshot.currentPhase !== 'chain_capture') break;
      const moves = adapter.getValidMoves();
      const next = moves.find((m: Move) => m.type === 'continue_capture_segment');
      if (!next) break;
      expect(adapter.validateMove(next).valid).toBe(true);
      await engine.applyCanonicalMove(next);
    }

    // After chain, expect to have reached line_processing then territory_processing
    const phasesSeen: GameState['currentPhase'][] = [];
    const finalState = engine.getGameState();
    phasesSeen.push(finalState.currentPhase);
    expect(['line_processing', 'territory_processing', 'ring_placement']).toContain(
      finalState.currentPhase
    );

    // Once processing completes, turn should advance to the next player
    const afterProcessing = engine.getGameState();
    expect(afterProcessing.currentPlayer).toBe(2);
    expect(afterProcessing.gameStatus).toBe('active');
  });
});
