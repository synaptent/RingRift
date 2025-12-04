import { processTurn, getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, Move } from '../../src/shared/types/game';
import { createSquareTerritoryRegionScenario } from '../helpers/squareTerritoryScenario';
import * as TerritoryAggregate from '../../src/shared/engine/aggregates/TerritoryAggregate';

describe('turnOrchestrator – skip_territory_processing', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('getValidMoves in territory_processing includes skip_territory_processing when regions exist and no elimination is pending', () => {
    const { initialState } = createSquareTerritoryRegionScenario('territory-skip-orchestrator');

    const regionMove: Move = {
      id: 'region-1',
      type: 'process_territory_region',
      player: initialState.currentPlayer,
      to: { x: 2, y: 2 },
      disconnectedRegions: [
        {
          spaces: [{ x: 2, y: 2 }],
          controllingPlayer: initialState.currentPlayer,
          isDisconnected: true,
        },
      ],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    jest
      .spyOn(TerritoryAggregate, 'enumerateProcessTerritoryRegionMoves')
      .mockReturnValueOnce([regionMove]);
    jest.spyOn(TerritoryAggregate, 'enumerateTerritoryEliminationMoves').mockReturnValueOnce([]);

    const moves = getValidMoves(initialState as GameState);

    const hasRegionMove = moves.some((m) => m.type === 'process_territory_region');
    const hasSkipMove = moves.some((m) => m.type === 'skip_territory_processing');

    expect(hasRegionMove).toBe(true);
    expect(hasSkipMove).toBe(true);
  });

  it('processTurn with skip_territory_processing leaves board unchanged and advances turn/phase', () => {
    const { initialState } = createSquareTerritoryRegionScenario('territory-skip-orchestrator-2');

    const state = initialState as GameState;

    const skipMove: Move = {
      id: 'skip-territory-test',
      type: 'skip_territory_processing',
      player: state.currentPlayer,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    const beforeStacks = new Map(state.board.stacks);
    const beforeCollapsed = new Map(state.board.collapsedSpaces);
    const beforeElims = { ...state.board.eliminatedRings };
    const beforePlayer = state.currentPlayer;
    const beforePhase = state.currentPhase;

    const result = processTurn(state, skipMove);
    const after = result.nextState as GameState;

    expect(after.board.stacks.size).toBe(beforeStacks.size);
    expect(after.board.collapsedSpaces.size).toBe(beforeCollapsed.size);
    expect(after.board.eliminatedRings).toEqual(beforeElims);

    expect(after.currentPlayer).not.toBe(beforePlayer);
    expect(after.currentPhase).not.toBe(beforePhase);
  });
});
