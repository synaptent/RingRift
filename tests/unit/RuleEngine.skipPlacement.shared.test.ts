import { RuleEngine } from '../../src/server/game/RuleEngine';
import type {
  BoardState,
  GameState,
  Move,
  Player,
  Position,
  BoardType,
  TimeControl,
} from '../../src/shared/types/game';
import { evaluateSkipPlacementEligibilityAggregate } from '../../src/shared/engine';

class FakeBoardManagerForSkipPlacement {
  constructor(public boardType: BoardType) {}

  isValidPosition(_pos: Position): boolean {
    return true;
  }

  isCollapsedSpace(_pos: Position, _board: BoardState): boolean {
    return false;
  }

  getMarker(_pos: Position, _board: BoardState): number | undefined {
    return undefined;
  }

  getAllPositions(): Position[] {
    return [];
  }

  findAllLines(_board: BoardState): Array<{ player: number; positions: Position[] }> {
    return [];
  }

  findAllTerritories(_player: number, _board: BoardState): any[] {
    return [];
  }

  findDisconnectedRegions(_board: BoardState, _player: number): any[] {
    return [];
  }
}

function createBaseRingPlacementState(boardType: BoardType = 'square8'): GameState {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'rapid' };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 3,
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
      ringsInHand: 3,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: boardType === 'square8' ? 8 : boardType === 'square19' ? 19 : 11,
    type: boardType,
  };

  const now = new Date();

  return {
    id: 'ruleengine-skip-placement-test',
    boardType,
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl,
    spectators: [],
    gameStatus: 'active',
    createdAt: now,
    lastMoveAt: now,
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0,
  };
}

describe('RuleEngine skip_placement validation parity with PlacementAggregate', () => {
  const boardType: BoardType = 'square8';

  function createRuleEngineAndState() {
    const boardManager = new FakeBoardManagerForSkipPlacement(boardType) as any;
    const engine = new RuleEngine(boardManager, boardType as any);
    const state = createBaseRingPlacementState(boardType);
    return { engine, state };
  }

  it('accepts skip_placement exactly when canonical eligibility deems it legal', () => {
    const { engine, state } = createRuleEngineAndState();

    const skipMove: Move = {
      id: 'skip-1',
      type: 'skip_placement',
      player: 1,
      from: undefined,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    const aggregateEligibility = evaluateSkipPlacementEligibilityAggregate(state, 1);
    // Be tolerant of legacy helper shapes while treating the aggregate as SSOT:
    // - aggregates/PlacementAggregate: { eligible: boolean }
    // - placementHelpers:            { canSkip: boolean }
    const aggregateFlag =
      (aggregateEligibility as any).eligible ?? (aggregateEligibility as any).canSkip ?? false;

    const ruleEngineValid = engine.validateMove(skipMove, state);

    // Backend semantics: skip_placement is legal iff the player has rings in
    // hand and the aggregate eligibility helper reports a legal skip. This
    // matches the P2.1 tightening (ringsInHand > 0, controlled stack, and a
    // legal movement/capture from some controlled stack).
    const player = state.players.find((p) => p.playerNumber === 1)!;
    const canonicalFlag = player.ringsInHand > 0 && aggregateFlag;

    expect(ruleEngineValid).toBe(canonicalFlag);
  });

  it('rejects skip_placement when aggregate says ineligible (no stacks or no actions)', () => {
    const { engine, state } = createRuleEngineAndState();

    state.board.stacks.clear();

    const skipMove: Move = {
      id: 'skip-2',
      type: 'skip_placement',
      player: 1,
      from: undefined,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    const aggregateEligibility = evaluateSkipPlacementEligibilityAggregate(state, 1);
    const aggregateFlag =
      (aggregateEligibility as any).eligible ?? (aggregateEligibility as any).canSkip ?? false;

    const ruleEngineValid = engine.validateMove(skipMove, state);

    // When the aggregate reports ineligible (e.g. no controlled stacks or no
    // legal movement/capture), the backend must also reject skip_placement.
    expect(aggregateFlag).toBe(false);
    expect(ruleEngineValid).toBe(false);
  });
});
