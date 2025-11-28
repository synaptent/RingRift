import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  RingStack,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
  Move,
} from '../../src/shared/types/game';
import {
  applySimpleMovement,
  SimpleMovementParams,
} from '../../src/shared/engine/aggregates/MovementAggregate';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import type { Player } from '../../src/shared/types/game';

/**
 * Sandbox-side tests for the "landing on your own marker eliminates your top ring"
 * rule. These mirror tests/unit/GameEngine.landingOnOwnMarker.test.ts but operate
 * on ClientSandboxEngine so that the GUI / local AI obey the same semantics.
 */

describe('ClientSandboxEngine landing on own marker eliminates top ring', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        if (anyChoice.type === 'capture_direction') {
          const cd = anyChoice as CaptureDirectionChoice;
          const options = cd.options || [];
          if (options.length === 0) {
            throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
          }

          // Deterministically pick the first option for reproducibility.
          const selected = options[0];
          return {
            choiceId: cd.id,
            playerNumber: cd.playerNumber,
            choiceType: cd.type,
            selectedOption: selected,
          } as PlayerChoiceResponseFor<TChoice>;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  it("eliminates the mover's top ring when a simple move lands on an own marker", async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPhase = 'movement';
    state.currentPlayer = 1;

    const board = state.board;
    const from: Position = { x: 1, y: 1 };
    const to: Position = { x: 3, y: 1 };

    // Attacking stack: Player 1, height 2.
    const rings = [1, 1];
    const stack: RingStack = {
      position: from,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    };
    board.stacks.set(positionToString(from), stack);

    // Own marker at the landing position.
    board.markers.set(positionToString(to), {
      player: 1,
      position: to,
      type: 'regular',
    });

    // Sanity: reset elimination counters.
    state.totalRingsEliminated = 0;
    state.board.eliminatedRings = {};
    const player1 = state.players.find((p) => p.playerNumber === 1)!;
    player1.eliminatedRings = 0;

    // Simulate the click sequence: select source, then click destination.
    await engineAny.handleMovementClick(from);
    await engineAny.handleMovementClick(to);

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    const stackAtFrom = finalBoard.stacks.get(positionToString(from));
    const stackAtTo = finalBoard.stacks.get(positionToString(to));
    const markerAtTo = finalBoard.markers.get(positionToString(to));

    expect(stackAtFrom).toBeUndefined();
    expect(stackAtTo).toBeDefined();

    // The mover started with height 2 and should lose exactly one ring
    // when landing on their own marker.
    expect(stackAtTo!.stackHeight).toBe(1);

    // One ring eliminated globally and credited to player 1.
    expect(finalState.totalRingsEliminated).toBe(1);
    expect(finalState.board.eliminatedRings[1]).toBe(1);
    expect(finalState.players.find((p) => p.playerNumber === 1)!.eliminatedRings).toBe(1);

    // The landing marker should have been removed.
    expect(markerAtTo).toBeUndefined();
  });

  it("eliminates the mover's top ring when an overtaking capture lands on an own marker", async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPhase = 'movement';
    state.currentPlayer = 1;

    const board = state.board;
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
      controllingPlayer: 1,
    };

    // Target: Player 2, height 1.
    const targetRings = [2];
    const targetStack: RingStack = {
      position: target,
      rings: targetRings,
      stackHeight: targetRings.length,
      capHeight: targetRings.length,
      controllingPlayer: 2,
    };

    board.stacks.set(positionToString(from), attacker);
    board.stacks.set(positionToString(target), targetStack);

    // Own marker at the landing position.
    board.markers.set(positionToString(landing), {
      player: 1,
      position: landing,
      type: 'regular',
    });

    state.totalRingsEliminated = 0;
    state.board.eliminatedRings = {};
    const player1 = state.players.find((p) => p.playerNumber === 1)!;
    player1.eliminatedRings = 0;

    // Drive a single capture segment corresponding to from -> target -> landing.
    await engineAny.performCaptureChain(from, target, landing, 1);

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    const stackAtFrom = finalBoard.stacks.get(positionToString(from));
    const stackAtTarget = finalBoard.stacks.get(positionToString(target));
    const stackAtLanding = finalBoard.stacks.get(positionToString(landing));
    const markerAtLanding = finalBoard.markers.get(positionToString(landing));

    expect(stackAtFrom).toBeUndefined();
    expect(stackAtTarget).toBeUndefined();
    expect(stackAtLanding).toBeDefined();

    // Attacker height 2 + target height 1 = 3 rings total, minus one
    // eliminated for landing on own marker => final height 2.
    expect(stackAtLanding!.stackHeight).toBe(2);

    expect(finalState.totalRingsEliminated).toBe(1);
    expect(finalState.board.eliminatedRings[1]).toBe(1);
    expect(finalState.players.find((p) => p.playerNumber === 1)!.eliminatedRings).toBe(1);

    // Landing marker should have been removed as part of the rule.
    expect(markerAtLanding).toBeUndefined();
  });
});

/**
 * Tests for the orchestrator path (MovementAggregate.applySimpleMovement).
 * This exercises the fix for the race condition where the landing marker
 * state was being checked AFTER applyMarkerEffectsAlongPathOnBoard deleted it.
 */
describe('MovementAggregate orchestrator path - landing on own marker', () => {
  // Helper to create a minimal test state
  function createTestState(): GameState {
    const players: Player[] = [
      {
        id: 'player-1',
        username: 'Player 1',
        type: 'human',
        playerNumber: 1,
        timeRemaining: 600000,
        isReady: true,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'player-2',
        username: 'Player 2',
        type: 'human',
        playerNumber: 2,
        timeRemaining: 600000,
        isReady: true,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const timeControl = {
      initialTime: 600,
      increment: 0,
      type: 'rapid' as const,
    };

    return createInitialGameState('test-game', 'square8', players, timeControl);
  }

  it('should eliminate bottom ring when landing on own marker via orchestrator path', () => {
    // Setup: Create initial state for a 2-player square8 game
    const state = createTestState();

    // Manually set up the board state for this test
    state.currentPhase = 'movement';
    state.currentPlayer = 1;
    state.gameStatus = 'active';

    const from: Position = { x: 2, y: 2 };
    const to: Position = { x: 4, y: 2 };
    const fromKey = positionToString(from);
    const toKey = positionToString(to);

    // Clear any default stacks/markers
    state.board.stacks.clear();
    state.board.markers.clear();

    // Player 1's stack at origin (height 2)
    // rings array in RingStack is top→bottom (rings[0] is top/controlling ring)
    const rings = [1, 1];
    const stack: RingStack = {
      position: from,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    };
    state.board.stacks.set(fromKey, stack);

    // Player 1's own marker at the destination
    state.board.markers.set(toKey, {
      player: 1,
      position: to,
      type: 'regular',
    });

    // Reset elimination counters
    state.totalRingsEliminated = 0;
    state.board.eliminatedRings = {};
    const player1 = state.players.find((p: Player) => p.playerNumber === 1)!;
    player1.eliminatedRings = 0;

    // Action: Call applySimpleMovement directly (orchestrator path)
    const params: SimpleMovementParams = {
      from,
      to,
      player: 1,
    };

    const result = applySimpleMovement(state, params);
    const finalState = result.nextState;
    const finalBoard = finalState.board;

    // Assert: Stack is no longer at origin
    const stackAtFrom = finalBoard.stacks.get(fromKey);
    expect(stackAtFrom).toBeUndefined();

    // Assert: Stack is now at destination with height reduced by 1
    const stackAtTo = finalBoard.stacks.get(toKey);
    expect(stackAtTo).toBeDefined();
    expect(stackAtTo!.stackHeight).toBe(1); // Started with 2, lost 1

    // Assert: Landing marker was removed
    const markerAtTo = finalBoard.markers.get(toKey);
    expect(markerAtTo).toBeUndefined();

    // Assert: Elimination counts updated correctly
    expect(result.eliminatedRingsByPlayer[1]).toBe(1);
    expect(finalState.totalRingsEliminated).toBe(1);
    expect(finalState.board.eliminatedRings[1]).toBe(1);

    // Assert: Player elimination count updated
    const finalPlayer1 = finalState.players.find((p: Player) => p.playerNumber === 1)!;
    expect(finalPlayer1.eliminatedRings).toBe(1);
  });

  it('should completely remove stack when height-1 stack lands on own marker', () => {
    // Setup: Create initial state for a 2-player square8 game
    const state = createTestState();

    state.currentPhase = 'movement';
    state.currentPlayer = 1;
    state.gameStatus = 'active';

    const from: Position = { x: 1, y: 1 };
    const to: Position = { x: 2, y: 1 };
    const fromKey = positionToString(from);
    const toKey = positionToString(to);

    // Clear any default stacks/markers
    state.board.stacks.clear();
    state.board.markers.clear();

    // Player 1's stack at origin (height 1 - will be eliminated entirely)
    const rings = [1];
    const stack: RingStack = {
      position: from,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    };
    state.board.stacks.set(fromKey, stack);

    // Player 1's own marker at the destination
    state.board.markers.set(toKey, {
      player: 1,
      position: to,
      type: 'regular',
    });

    // Reset elimination counters
    state.totalRingsEliminated = 0;
    state.board.eliminatedRings = {};

    // Action: Call applySimpleMovement directly (orchestrator path)
    const params: SimpleMovementParams = {
      from,
      to,
      player: 1,
    };

    const result = applySimpleMovement(state, params);
    const finalState = result.nextState;
    const finalBoard = finalState.board;

    // Assert: Stack at origin is gone
    expect(finalBoard.stacks.get(fromKey)).toBeUndefined();

    // Assert: Stack at destination is also gone (completely eliminated)
    expect(finalBoard.stacks.get(toKey)).toBeUndefined();

    // Assert: Landing marker was removed
    expect(finalBoard.markers.get(toKey)).toBeUndefined();

    // Assert: One ring was eliminated
    expect(result.eliminatedRingsByPlayer[1]).toBe(1);
    expect(finalState.totalRingsEliminated).toBe(1);
  });

  it('should NOT eliminate rings when landing on empty space', () => {
    // Setup: Create initial state
    const state = createTestState();

    state.currentPhase = 'movement';
    state.currentPlayer = 1;
    state.gameStatus = 'active';

    const from: Position = { x: 2, y: 2 };
    const to: Position = { x: 4, y: 2 };
    const fromKey = positionToString(from);
    const toKey = positionToString(to);

    // Clear any default stacks/markers
    state.board.stacks.clear();
    state.board.markers.clear();

    // Player 1's stack at origin (height 2)
    const rings = [1, 1];
    const stack: RingStack = {
      position: from,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    };
    state.board.stacks.set(fromKey, stack);

    // NO marker at destination (empty space)

    // Reset counters
    state.totalRingsEliminated = 0;
    state.board.eliminatedRings = {};

    // Action
    const params: SimpleMovementParams = {
      from,
      to,
      player: 1,
    };

    const result = applySimpleMovement(state, params);
    const finalState = result.nextState;
    const finalBoard = finalState.board;

    // Assert: Stack moved without height change
    expect(finalBoard.stacks.get(fromKey)).toBeUndefined();
    const stackAtTo = finalBoard.stacks.get(toKey);
    expect(stackAtTo).toBeDefined();
    expect(stackAtTo!.stackHeight).toBe(2); // No reduction

    // Assert: No elimination
    expect(result.eliminatedRingsByPlayer).toEqual({});
    expect(finalState.totalRingsEliminated).toBe(0);
  });
});
