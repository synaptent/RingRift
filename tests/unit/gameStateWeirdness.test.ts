import type { GameState, GameResult, BoardState, Player } from '../../src/shared/types/game';
import { getWeirdStateBanner } from '../../src/client/utils/gameStateWeirdness';

jest.mock('../../src/shared/engine/globalActions', () => ({
  __esModule: true,
  computeGlobalLegalActionsSummary: jest.fn(),
  isANMState: jest.fn(),
}));

const { computeGlobalLegalActionsSummary, isANMState } =
  require('../../src/shared/engine/globalActions') as {
    computeGlobalLegalActionsSummary: jest.Mock;
    isANMState: jest.Mock;
  };

function createBaseGameState(overrides: Partial<GameState> = {}): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  const base: GameState = {
    id: 'test-game',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };

  return { ...base, ...overrides };
}

describe('getWeirdStateBanner', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    isANMState.mockReturnValue(false);
    computeGlobalLegalActionsSummary.mockReturnValue({
      hasTurnMaterial: false,
      hasForcedEliminationAction: false,
      hasGlobalPlacementAction: false,
      hasPhaseLocalInteractiveMove: false,
    });
  });

  it('returns none for a normal active state when helpers report no weirdness', () => {
    const state = createBaseGameState();

    const result = getWeirdStateBanner(state);

    expect(result).toEqual({ type: 'none' });
  });

  it('classifies structural stalemate when game has completed with reason game_completed', () => {
    const state = createBaseGameState({ gameStatus: 'completed' });

    const victory: GameResult = {
      winner: 1,
      reason: 'game_completed',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };

    const result = getWeirdStateBanner(state, { victoryState: victory });

    expect(result).toEqual({
      type: 'structural-stalemate',
      winner: 1,
      reason: 'game_completed',
    });
  });

  it('classifies Active–No–Moves in movement-like phases via isANMState', () => {
    const state = createBaseGameState({
      currentPhase: 'movement',
      currentPlayer: 2,
    });

    isANMState.mockReturnValue(true);

    const result = getWeirdStateBanner(state);

    expect(result).toEqual({ type: 'active-no-moves-movement', playerNumber: 2 });
  });

  it('classifies Active–No–Moves during line_processing as active-no-moves-line', () => {
    const state = createBaseGameState({
      currentPhase: 'line_processing',
      currentPlayer: 1,
    });

    isANMState.mockReturnValue(true);

    const result = getWeirdStateBanner(state);

    expect(result).toEqual({ type: 'active-no-moves-line', playerNumber: 1 });
  });

  it('classifies Active–No–Moves during territory_processing as active-no-moves-territory', () => {
    const state = createBaseGameState({
      currentPhase: 'territory_processing',
      currentPlayer: 1,
    });

    isANMState.mockReturnValue(true);

    const result = getWeirdStateBanner(state);

    expect(result).toEqual({ type: 'active-no-moves-territory', playerNumber: 1 });
  });

  it('classifies forced-elimination when summary exposes only FE global action', () => {
    const state = createBaseGameState({
      currentPhase: 'movement',
      currentPlayer: 1,
    });

    isANMState.mockReturnValue(false);
    computeGlobalLegalActionsSummary.mockReturnValue({
      hasTurnMaterial: true,
      hasForcedEliminationAction: true,
      hasGlobalPlacementAction: false,
      hasPhaseLocalInteractiveMove: false,
    });

    const result = getWeirdStateBanner(state);

    expect(result).toEqual({ type: 'forced-elimination', playerNumber: 1 });
  });

  it('does not classify forced-elimination when placements or phase-local moves exist', () => {
    const state = createBaseGameState({
      currentPhase: 'movement',
      currentPlayer: 1,
    });

    isANMState.mockReturnValue(false);
    computeGlobalLegalActionsSummary.mockReturnValue({
      hasTurnMaterial: true,
      hasForcedEliminationAction: true,
      hasGlobalPlacementAction: true,
      hasPhaseLocalInteractiveMove: true,
    });

    const result = getWeirdStateBanner(state);

    expect(result).toEqual({ type: 'none' });
  });
});
