import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { BoardType, GameState, Move, Position, positionToString } from '../../src/shared/types/game';
import { createTestBoard, createTestGameState, createTestPlayer, addMarker } from '../utils/fixtures';
import { enumerateLegalRingPlacements, PlacementBoardView } from '../../src/client/sandbox/sandboxPlacement';

describe('Placement parity between backend RuleEngine and sandbox helpers', () => {
  const boardType: BoardType = 'square8';

  function createBackendStateWithMarkerAt(
    pos: Position,
    currentPlayer: number
  ): { state: GameState; manager: BoardManager; engine: RuleEngine } {
    const board = createTestBoard(boardType);
    addMarker(board, pos, 1);
    const players = [
      createTestPlayer(1, { type: 'human', ringsInHand: 10 }),
      createTestPlayer(2, { type: 'human', ringsInHand: 10 }),
    ];
    const state = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer,
      currentPhase: 'ring_placement',
    });

    const manager = new BoardManager(boardType);
    const engine = new RuleEngine(manager, boardType as any);

    return { state, manager, engine };
  }

  it('backend and sandbox agree on legal placement squares in a marker scenario (no placement onto markers)', () => {
    const markerPos: Position = { x: 1, y: 5 };
    const currentPlayer = 2;

    const { state, manager, engine } = createBackendStateWithMarkerAt(markerPos, currentPlayer);
    const board = state.board;

    const backendMoves: Move[] = engine
      .getValidMoves(state)
      .filter((m) => m.type === 'place_ring' && m.player === currentPlayer);

    const backendDestinations = new Set(
      backendMoves
        .map((m) => (m.to ? positionToString(m.to) : null))
        .filter((k): k is string => k !== null)
    );

    const view: PlacementBoardView = {
      isValidPosition: (pos) => manager.isValidPosition(pos),
      isCollapsedSpace: (pos, b) => {
        const key = positionToString(pos);
        return (b ?? board).collapsedSpaces.has(key);
      },
      getMarkerOwner: (pos, b) => {
        const key = positionToString(pos);
        const markers = (b ?? board).markers;
        const marker = markers.get(key);
        return marker?.player;
      },
    };

    const sandboxPositions = enumerateLegalRingPlacements(boardType, board, currentPlayer, view);
    const sandboxDestinations = new Set(
      sandboxPositions.map((pos) => positionToString(pos))
    );

    // Sets should be identical.
    expect(sandboxDestinations).toEqual(backendDestinations);

    const markerKey = positionToString(markerPos);
    expect(backendDestinations.has(markerKey)).toBe(false);
    expect(sandboxDestinations.has(markerKey)).toBe(false);
  });
});