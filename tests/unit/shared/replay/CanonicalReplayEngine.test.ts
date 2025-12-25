import { CanonicalReplayEngine } from '../../../../src/shared/replay';
import type { Move } from '../../../../src/shared/types/game';

describe('CanonicalReplayEngine', () => {
  it('applies a no_placement_action in ring_placement and advances turn', async () => {
    const engine = new CanonicalReplayEngine({
      gameId: 'test-game',
      boardType: 'square8',
      numPlayers: 2,
    });

    const move: Move = {
      id: 'm1',
      type: 'no_placement_action',
      player: 1,
      to: { x: 0, y: 0, z: null },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = await engine.applyMove(move);

    expect(result.success).toBe(true);
    expect(result.state.currentPlayer).toBe(1);
    expect(result.state.currentPhase).toBe('movement');
    // CanonicalReplayEngine uses 'active' for parity with Python's selfplay state
    expect(result.state.gameStatus).toBe('active');
  });
});
