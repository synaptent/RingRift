import {
  GameState,
  PlaceRingAction,
  MoveStackAction,
  ProcessLineAction,
  ChooseLineRewardAction,
  EliminateStackAction,
  ProcessTerritoryAction,
} from '../../src/shared/engine/types';
import {
  BoardType,
  Player,
  TimeControl,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { mutatePlacement } from '../../src/shared/engine/mutators/PlacementMutator';
import { mutateMovement } from '../../src/shared/engine/mutators/MovementMutator';
import {
  mutateProcessLine,
  mutateChooseLineReward,
} from '../../src/shared/engine/mutators/LineMutator';
import {
  mutateProcessTerritory,
  mutateEliminateStack,
} from '../../src/shared/engine/mutators/TerritoryMutator';
import { mutateTurnChange, mutatePhaseChange } from '../../src/shared/engine/mutators/TurnMutator';

describe('Shared engine mutators – basic invariants and S-invariant', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function snapshotS(state: GameState): number {
    // computeProgressSnapshot is typed against the legacy/shared GameState type
    // in src/shared/types/game. For mutator-level invariant checks we only
    // rely on board geometry and elimination counters, so a cast is sufficient.
    return computeProgressSnapshot(state as any).S;
  }

  it('mutatePlacement keeps S-invariant non-decreasing and stack/rings consistent', () => {
    const initial: GameState = createInitialGameState(
      'mutator-placement',
      boardType,
      players,
      timeControl
    );
    const beforeS = snapshotS(initial);

    const action: PlaceRingAction = {
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 1,
    };

    const after = mutatePlacement(initial, action);
    const afterS = snapshotS(after);

    // Placement affects only ringsInHand and stacks, not markers/collapsed/eliminated.
    expect(afterS).toBeGreaterThanOrEqual(beforeS);

    const stack = after.board.stacks.get('0,0');
    expect(stack).toBeDefined();
    if (stack) {
      expect(stack.stackHeight).toBe(stack.rings.length);
      expect(stack.stackHeight).toBe(1);
    }

    const p1Before = initial.players.find((p) => p.playerNumber === 1)!;
    const p1After = after.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.ringsInHand).toBeLessThanOrEqual(p1Before.ringsInHand);
    expect(p1After.ringsInHand).toBe(p1Before.ringsInHand - 1);
  });

  it('mutateMovement preserves S-invariant and basic stack invariants', () => {
    const base: GameState = createInitialGameState('mutator-move', boardType, players, timeControl);

    const place: PlaceRingAction = {
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 1,
    };
    const withStack = mutatePlacement(base, place);
    const beforeS = snapshotS(withStack);

    const move: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
    };
    const after = mutateMovement(withStack, move);
    const afterS = snapshotS(after);

    expect(afterS).toBeGreaterThanOrEqual(beforeS);

    const stack = after.board.stacks.get('0,1');
    expect(stack).toBeDefined();
    if (stack) {
      expect(stack.stackHeight).toBe(stack.rings.length);
    }

    const originMarker = after.board.markers.get('0,0');
    expect(originMarker?.player).toBe(1);
  });

  it('mutateProcessLine increases or preserves S-invariant and clears processed line', () => {
    const state: GameState = createInitialGameState(
      'mutator-line',
      boardType,
      players,
      timeControl
    );

    // Seed a simple exact-length line for player 1 with markers only.
    const linePositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
    ];

    for (const pos of linePositions) {
      const key = positionToString(pos);
      (state.board.markers as any).set(key, { player: 1, position: pos, type: 'regular' });
    }

    (state.board as any).formedLines = [
      {
        player: 1,
        positions: linePositions,
        length: linePositions.length,
      },
    ];

    const beforeS = snapshotS(state);

    const action: ProcessLineAction = {
      type: 'PROCESS_LINE',
      playerId: 1,
      lineIndex: 0,
    };

    const after = mutateProcessLine(state, action);
    const afterS = snapshotS(after);

    expect(afterS).toBeGreaterThanOrEqual(beforeS);

    // Processed line removed.
    expect(after.board.formedLines.length).toBe(0);

    // All positions in the line are now collapsed territory for player 1.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(after.board.markers.get(key)).toBeUndefined();
      expect(after.board.collapsedSpaces.get(key)).toBe(1);
    }
  });

  it('mutateChooseLineReward (MINIMUM_COLLAPSE) leaves S-invariant non-decreasing', () => {
    const state: GameState = createInitialGameState(
      'mutator-line-reward',
      boardType,
      players,
      timeControl
    );

    const linePositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 },
    ];

    (state.board as any).formedLines = [
      {
        player: 1,
        positions: linePositions,
        length: linePositions.length,
      },
    ];

    const beforeS = snapshotS(state);

    const minSubset = linePositions.slice(0, 4);
    const action: ChooseLineRewardAction = {
      type: 'CHOOSE_LINE_REWARD',
      playerId: 1,
      lineIndex: 0,
      selection: 'MINIMUM_COLLAPSE',
      collapsedPositions: minSubset,
    };

    const after = mutateChooseLineReward(state, action);
    const afterS = snapshotS(after);

    expect(afterS).toBeGreaterThanOrEqual(beforeS);

    // Only subset positions should be collapsed.
    for (const pos of minSubset) {
      const key = positionToString(pos);
      expect(after.board.collapsedSpaces.get(key)).toBe(1);
    }
    expect(after.board.collapsedSpaces.get('4,0')).toBeUndefined();
  });

  it('mutateEliminateStack eliminates the full cap and updates S-invariant / elimination counts', () => {
    const state: GameState = createInitialGameState(
      'mutator-eliminate',
      boardType,
      players,
      timeControl
    );

    // Seed a simple 2-ring *pure cap* stack for player 1. Under the
    // Q23 / self-elimination semantics used by the sandbox and
    // ClientSandboxEngine, an explicit elimination removes the entire
    // cap, which in this case is the whole stack.
    (state.board.stacks as any).set('0,0', {
      position: { x: 0, y: 0 },
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    });
    (state.board.eliminatedRings as any)[1] = 0;
    const p1 = state.players.find((p) => p.playerNumber === 1)!;
    (p1 as any).eliminatedRings = 0;

    const beforeS = snapshotS(state);

    const action: EliminateStackAction = {
      type: 'ELIMINATE_STACK',
      playerId: 1,
      stackPosition: { x: 0, y: 0 },
    };

    const after = mutateEliminateStack(state, action);
    const afterS = snapshotS(after);

    // Eliminating a cap increases E and therefore S, or at least leaves
    // it non-decreasing if additional invariants are introduced later.
    expect(afterS).toBeGreaterThanOrEqual(beforeS);

    // Because the stack was a pure cap for player 1, the entire stack
    // should be gone after elimination.
    const stack = after.board.stacks.get('0,0');
    expect(stack).toBeUndefined();

    const p1After = after.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.eliminatedRings).toBe(2);
    expect(after.board.eliminatedRings[1]).toBe(2);
  });

  it('mutateProcessTerritory does not decrease S-invariant when marking a region as connected', () => {
    const state: GameState = createInitialGameState(
      'mutator-territory',
      boardType,
      players,
      timeControl
    );

    (state.board.territories as any).set('region-1', {
      id: 'region-1',
      controllingPlayer: 1,
      isDisconnected: true,
      spaces: [{ x: 0, y: 0 }],
    });

    // Seed a collapsed space to make S-invariant meaningful.
    (state.board.collapsedSpaces as any).set('0,0', 1);

    const beforeS = snapshotS(state);

    const action: ProcessTerritoryAction = {
      type: 'PROCESS_TERRITORY',
      playerId: 1,
      regionId: 'region-1',
    };

    const after = mutateProcessTerritory(state, action);
    const afterS = snapshotS(after);

    expect(afterS).toBeGreaterThanOrEqual(beforeS);

    const kept = (after.board.territories as any).get('region-1');
    expect(kept).toBeDefined();
    expect(kept.isDisconnected).toBe(false);
  });

  it('mutateTurnChange keeps S-invariant constant and advances currentPlayer', () => {
    const state: GameState = createInitialGameState(
      'mutator-turn',
      boardType,
      players,
      timeControl
    );

    const beforeS = snapshotS(state);
    const beforePlayer = state.currentPlayer;

    const after = mutateTurnChange(state);
    const afterS = snapshotS(after);

    expect(afterS).toBe(beforeS);
    expect(after.currentPlayer).not.toBe(beforePlayer);
  });

  it('mutatePhaseChange keeps S-invariant constant and updates phase', () => {
    const state: GameState = createInitialGameState(
      'mutator-phase',
      boardType,
      players,
      timeControl
    );

    const beforeS = snapshotS(state);
    const after = mutatePhaseChange(state, 'movement');

    const afterS = snapshotS(after);
    expect(afterS).toBe(beforeS);
    expect(after.currentPhase).toBe('movement');
  });
});
