import type { GameState, Move } from '../../src/shared/types/game';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { createSquare19TwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';

describe('turnOrchestrator territorySkip → ForcedElimination regression', () => {
  it('does not end the game immediately after skip_territory_processing when forced elimination should occur', () => {
    // Historical parity bug:
    // - TS previously evaluated victory + returned currentPhase='game_over' immediately after
    //   skip_territory_processing, even when the canonical post-territory progression should
    //   enter forced_elimination first.
    // - This was fixed by mirroring the no_territory_action forced-elimination gate in
    //   turnOrchestrator.ts (see switch case 'skip_territory_processing').
    const { initialState } = createSquare19TwoRegionTerritoryScenario(
      'regression-territory-skip-to-forced-elimination'
    );

    const state: GameState = {
      ...initialState,
      // Ensure the turn has had no real actions (no prior moves for this player).
      moveHistory: [],
      history: [],
      currentPlayer: 1,
      currentPhase: 'territory_processing',
      gameStatus: 'active',
      // Make victory evaluation trivially true *if* it runs before forced elimination.
      // The regression is about ordering: forced_elimination must be surfaced first.
      players: initialState.players.map((p) =>
        p.playerNumber === 1 ? { ...p, eliminatedRings: initialState.victoryThreshold } : p
      ),
    };

    const skipMove: Move = {
      id: 'skip-territory-1',
      type: 'skip_territory_processing',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(0),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    const result = processTurn(state, skipMove);

    expect(result.nextState.gameStatus).toBe('active');
    expect(result.nextState.currentPlayer).toBe(1);
    expect(result.nextState.currentPhase).toBe('forced_elimination');

    // Must surface an explicit forced-elimination decision instead of silently
    // terminating or rotating the turn.
    expect(result.victoryResult).toBeUndefined();
    expect(result.pendingDecision?.type).toBe('elimination_target');
    expect(result.pendingDecision?.options.length).toBeGreaterThan(0);
    expect(result.pendingDecision?.options.every((m) => m.type === 'forced_elimination')).toBe(
      true
    );
  });
});
