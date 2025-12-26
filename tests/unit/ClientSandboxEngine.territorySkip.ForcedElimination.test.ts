import type { Move } from '../../src/shared/types/game';
import { serializeGameState } from '../../src/shared/engine/contracts/serialization';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import { createSquare19TwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';

// Optional parity-style regression:
// Ensure the sandbox adapter (which wraps the shared orchestrator) observes the
// same territory_processing → forced_elimination transition for
// skip_territory_processing.

describe('ClientSandboxEngine territorySkip → ForcedElimination parity regression', () => {
  it('previewMove shows territory_processing → forced_elimination (not game_over) after skip_territory_processing', () => {
    const { initialState } = createSquare19TwoRegionTerritoryScenario(
      'sandbox-regression-territory-skip-to-forced-elimination'
    );

    const state = {
      ...initialState,
      moveHistory: [],
      history: [],
      currentPlayer: 1,
      currentPhase: 'territory_processing',
      gameStatus: 'active',
      players: initialState.players.map((p) =>
        p.playerNumber === 1 ? { ...p, eliminatedRings: initialState.victoryThreshold } : p
      ),
    };

    const interactionHandler = {
      requestChoice: async (choice: any) => ({ selectedOption: choice.options?.[0] }),
    } as any;

    const engine = new ClientSandboxEngine({
      config: {
        boardType: 'square19',
        numPlayers: 2,
        playerKinds: ['human', 'human'],
      },
      interactionHandler,
      traceMode: true,
    });

    engine.initFromSerializedState(
      serializeGameState(state as any),
      ['human', 'human'],
      interactionHandler
    );

    // Intentionally reach into the adapter for a pure "what would happen" check.
    // This avoids auto-resolving decisions during processMove().
    const adapter = (engine as any).getOrchestratorAdapter();

    const skipMove: Move = {
      id: 'skip-territory-1',
      type: 'skip_territory_processing',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(0),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    const preview = adapter.previewMove(skipMove);

    expect(preview.valid).toBe(true);
    expect(preview.nextState.gameStatus).toBe('active');
    expect(preview.nextState.currentPhase).toBe('forced_elimination');
  });
});
