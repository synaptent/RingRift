import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  Player,
  Position,
  RingStack,
  TimeControl
} from '../../src/shared/types/game';

/**
 * Tests for the "landing on your own marker eliminates your top ring" rule
 * in the backend GameEngine. These focus on the concrete board mutations
 * performed by GameEngine.applyMove / performOvertakingCapture rather than
 * RuleEngine validation.
 */

describe('GameEngine landing on own marker eliminates top ring', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  function createEngine(): any {
    // GameEngine constructor will reassign playerNumber/timeRemaining but
    // that is fine for these tests.
    const engine = new GameEngine('own-marker-rule', boardType, basePlayers, timeControl, false);
    return engine as any;
  }

  it('eliminates the mover\'s top ring when a non-capture move lands on an own marker', () => {
    const engineAny = createEngine();
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    const from: Position = { x: 1, y: 1 };
    const to: Position = { x: 3, y: 1 };

    // Attacking stack: Player 1, height 2.
    const rings = [1, 1];
    const stack: RingStack = {
      position: from,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1
    };
    boardManager.setStack(from, stack, gameState.board);

    // Own marker at the landing position.
    boardManager.setMarker(to, 1, gameState.board);

    // Sanity: no rings eliminated yet.
    gameState.totalRingsEliminated = 0;
    gameState.board.eliminatedRings = {};
    const player1 = gameState.players.find((p: Player) => p.playerNumber === 1)!;
    player1.eliminatedRings = 0;

    const move = {
      id: 'm-own-marker-move',
      type: 'move_ring' as const,
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    // Bypass RuleEngine and apply the move directly to test board mutations.
    engineAny.applyMove(move);

    // Re-read gameState after applyMove (it's reassigned internally)
    const updatedState = engineAny.gameState as any;

    const stackAtFrom = boardManager.getStack(from, updatedState.board);
    const stackAtTo = boardManager.getStack(to, updatedState.board) as RingStack | undefined;
    const markerAtTo = boardManager.getMarker(to, updatedState.board);

    expect(stackAtFrom).toBeUndefined();
    expect(stackAtTo).toBeDefined();

    // The mover started with height 2 and should lose exactly one ring
    // when landing on their own marker.
    expect(stackAtTo!.stackHeight).toBe(1);

    // One ring eliminated globally and credited to player 1.
    const updatedPlayer1 = updatedState.players.find((p: Player) => p.playerNumber === 1)!;
    expect(updatedState.totalRingsEliminated).toBe(1);
    expect(updatedState.board.eliminatedRings[1]).toBe(1);
    expect(updatedPlayer1.eliminatedRings).toBe(1);

    // The landing marker should have been removed.
    expect(markerAtTo).toBeUndefined();
  });

  it('eliminates the mover\'s top ring when an overtaking capture lands on an own marker', () => {
    const engineAny = createEngine();
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    const from: Position = { x: 1, y: 1 };
    const target: Position = { x: 2, y: 1 };
    const landing: Position = { x: 3, y: 1 };

    // Attacker: Player 1, height 2.
    const attackerRings = [1, 1];
    const attacker: RingStack = {
      position: from,
      rings: attackerRings,
      stackHeight: attackerRings.length,
      capHeight: attackerRings.length,
      controllingPlayer: 1
    };

    // Target: Player 2, height 1.
    const targetRings = [2];
    const targetStack: RingStack = {
      position: target,
      rings: targetRings,
      stackHeight: targetRings.length,
      capHeight: targetRings.length,
      controllingPlayer: 2
    };

    boardManager.setStack(from, attacker, gameState.board);
    boardManager.setStack(target, targetStack, gameState.board);

    // Own marker at the landing position.
    boardManager.setMarker(landing, 1, gameState.board);

    gameState.totalRingsEliminated = 0;
    gameState.board.eliminatedRings = {};
    const player1 = gameState.players.find((p: Player) => p.playerNumber === 1)!;
    player1.eliminatedRings = 0;

    const move = {
      id: 'c-own-marker-capture',
      type: 'overtaking_capture' as const,
      player: 1,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    engineAny.applyMove(move);

    // Re-read gameState after applyMove (it's reassigned internally)
    const updatedState = engineAny.gameState as any;

    const stackAtFrom = boardManager.getStack(from, updatedState.board);
    const stackAtTarget = boardManager.getStack(target, updatedState.board);
    const stackAtLanding = boardManager.getStack(landing, updatedState.board) as RingStack | undefined;
    const markerAtLanding = boardManager.getMarker(landing, updatedState.board);

    expect(stackAtFrom).toBeUndefined();
    expect(stackAtTarget).toBeUndefined();
    expect(stackAtLanding).toBeDefined();

    // Attacker height 2 + target height 1 = 3 rings total, minus one
    // eliminated for landing on own marker => final height 2.
    expect(stackAtLanding!.stackHeight).toBe(2);

    const updatedPlayer1 = updatedState.players.find((p: Player) => p.playerNumber === 1)!;
    expect(updatedState.totalRingsEliminated).toBe(1);
    expect(updatedState.board.eliminatedRings[1]).toBe(1);
    expect(updatedPlayer1.eliminatedRings).toBe(1);

    // Landing marker should have been removed as part of the rule.
    expect(markerAtLanding).toBeUndefined();
  });
});
