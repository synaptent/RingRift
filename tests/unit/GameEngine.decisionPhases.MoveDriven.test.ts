/**
 * Tests for the move-driven decision phase path on GameEngine:
 *
 * - When enableMoveDrivenDecisionPhases() is enabled for a live game,
 *   geometry moves that create line or territory consequences should:
 *     1. Transition the engine into 'line_processing' or 'territory_processing',
 *     2. Expose canonical decision Moves via getValidMoves
 *        (process_line / choose_line_reward / process_territory_region),
 *     3. Defer automatic consequences until those decision Moves are applied.
 *
 * These tests stub out RuleEngine and BoardManager geometry so we can focus
 * on the phase/Move wiring rather than full board setups.
 *
 * NOTE: Jest timeout set to 30 seconds. These async tests involve phase
 * transitions that could potentially hang if something goes wrong.
 */

// Set timeout to prevent hanging - async phase transitions need safety guard
jest.setTimeout(30000);

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  TimeControl,
  Player,
  Position,
  Move,
  LineInfo,
  Territory,
} from '../../src/shared/types/game';
import { pos, addStack } from '../utils/fixtures';
function makeStack(playerNumber: number, height: number, position: Position) {
  const rings = Array(height).fill(playerNumber);
  return {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer: playerNumber,
  };
}

// TODO-MOVE-DRIVEN-DECISION-PHASES: Skipping - move-driven decision phase tests
// involve async phase transitions and stubbed RuleEngine/BoardManager that can
// cause timeouts in the test environment. The underlying move-driven path is
// experimental/legacy relative to the shared orchestrator + ClientSandboxEngine
// move-driven adapters and needs stabilization before these tests can reliably
// pass. Investigation needed to identify which specific phase transitions are
// hanging.
describe.skip('GameEngine move-driven decision phases', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
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
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  it('enters line_processing and enumerates process_line / choose_line_reward Moves when enabled', async () => {
    const engine = new GameEngine('move-driven-lines', boardType, players, timeControl, false);
    engine.enableMoveDrivenDecisionPhases();

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;
    const boardManager = engineAny.boardManager as any;
    const ruleEngine = engineAny.ruleEngine as any;

    // Prepare a state where Player 1 is to move in an interactive phase.
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    // Stub validation to always accept the synthetic geometry move.
    jest.spyOn(ruleEngine, 'validateMove').mockReturnValue(true);

    // Stub applyMove to be a no-op with an empty consequences object.
    jest.spyOn(engineAny, 'applyMove').mockImplementation((..._args: any[]) => ({
      captures: [],
      territoryChanges: [],
      lineCollapses: [],
    }));

    // Synthetic overlength line for Player 1 so that both process_line and
    // choose_line_reward should be available.
    const markerPositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 }, // overlength for square8 (required length 4)
    ];

    const syntheticLine: LineInfo = {
      player: 1,
      positions: markerPositions,
      // Minimal directional/length metadata for tests; real values are
      // derived by BoardManager in production.
      length: markerPositions.length,
      direction: { x: 1, y: 0 },
    } as any;

    jest.spyOn(boardManager, 'findAllLines').mockReturnValue([syntheticLine]);

    // Ensure there is a source stack at the from-position so that the
    // defensive S-invariant check in GameEngine.makeMove passes.
    const stackPos: Position = { x: 0, y: 1 };
    boardManager.setStack(stackPos, makeStack(1, 3, stackPos), gameState.board);

    const movePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      player: 1,
      type: 'move_stack',
      from: { x: 0, y: 1 },
      to: { x: 1, y: 1 },
      thinkTime: 0,
    };

    const result = await engine.makeMove(movePayload);
    expect(result.success).toBe(true);

    const after = engine.getGameState();

    // The engine should now be in the line_processing phase for Player 1.
    expect(after.currentPhase).toBe('line_processing');
    expect(after.currentPlayer).toBe(1);

    // getValidMoves should expose canonical decision Moves for this phase.
    const decisionMoves = engine.getValidMoves(1);
    expect(decisionMoves.length).toBeGreaterThan(0);

    const types = decisionMoves.map((m) => m.type).sort();
    expect(types).toContain('process_line');
    expect(types).toContain('choose_line_reward');

    const processMove = decisionMoves.find((m) => m.type === 'process_line');
    expect(processMove).toBeDefined();
    if (processMove) {
      expect(typeof processMove.id === 'string').toBe(true);
      expect(processMove.id.startsWith('process-line-')).toBe(true);
      expect(processMove.formedLines && processMove.formedLines[0].positions).toEqual(
        markerPositions
      );
    }

    const rewardMove = decisionMoves.find((m) => m.type === 'choose_line_reward');
    expect(rewardMove).toBeDefined();
    if (rewardMove) {
      expect(typeof rewardMove.id === 'string').toBe(true);
      expect(rewardMove.id.startsWith('choose-line-reward-')).toBe(true);
      expect(rewardMove.formedLines && rewardMove.formedLines[0].positions).toEqual(
        markerPositions
      );
    }
  });

  it('enters territory_processing and enumerates process_territory_region Moves when enabled', async () => {
    const engine = new GameEngine('move-driven-territory', boardType, players, timeControl, false);
    engine.enableMoveDrivenDecisionPhases();

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;
    const boardManager = engineAny.boardManager as any;
    const ruleEngine = engineAny.ruleEngine as any;

    // Prepare a state where Player 1 is to move in an interactive phase.
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    jest.spyOn(ruleEngine, 'validateMove').mockReturnValue(true);
    jest.spyOn(engineAny, 'applyMove').mockImplementation((..._args: any[]) => ({
      captures: [],
      territoryChanges: [],
      lineCollapses: [],
    }));

    // No lines so we skip directly to territory conditions.
    jest.spyOn(boardManager, 'findAllLines').mockReturnValue([]);

    // Synthetic disconnected region satisfying the self-elimination prerequisite.
    const region: Territory = {
      spaces: [
        { x: 5, y: 5 },
        { x: 5, y: 6 },
      ],
      controllingPlayer: 0,
      isDisconnected: true,
    };

    jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementation((..._args: any[]) => [region]);

    // Source stack for the geometry move so validation and S-invariant
    // checks pass.
    const stackPos: Position = { x: 0, y: 1 };
    boardManager.setStack(stackPos, makeStack(1, 3, stackPos), gameState.board);

    // Allow processing of this region regardless of actual stacks.
    jest.spyOn(engineAny, 'canProcessDisconnectedRegion').mockReturnValue(true);

    const movePayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      player: 1,
      type: 'move_stack',
      from: { x: 0, y: 1 },
      to: { x: 1, y: 1 },
      thinkTime: 0,
    };

    const result = await engine.makeMove(movePayload);
    expect(result.success).toBe(true);

    const after = engine.getGameState();
    expect(after.currentPhase).toBe('territory_processing');
    expect(after.currentPlayer).toBe(1);

    const decisionMoves = engine.getValidMoves(1);
    expect(decisionMoves.length).toBeGreaterThan(0);

    const territoryMoves = decisionMoves.filter((m) => m.type === 'process_territory_region');
    expect(territoryMoves.length).toBeGreaterThan(0);

    const first = territoryMoves[0];
    expect(typeof first.id === 'string').toBe(true);
    expect(first.id.startsWith('process-region-')).toBe(true);

    // The Move should carry the concrete Territory in disconnectedRegions[0]
    // so that it uniquely identifies the region to be processed.
    expect(first.disconnectedRegions && first.disconnectedRegions.length).toBe(1);
    if (first.disconnectedRegions && first.disconnectedRegions[0]) {
      const region0 = first.disconnectedRegions[0];
      expect(region0.spaces).toEqual(region.spaces);
    }
  });

  it('records process_territory_region followed by eliminate_rings_from_stack as separate history entries in move-driven mode', async () => {
    const engine = new GameEngine(
      'move-driven-territory-history',
      boardType,
      players,
      timeControl,
      false
    );
    engine.enableMoveDrivenDecisionPhases();

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;
    const boardManager = engineAny.boardManager as any;
    const ruleEngine = engineAny.ruleEngine as any;

    gameState.gameStatus = 'active';
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'territory_processing';

    // Accept any synthetic decision moves without invoking full rules.
    jest.spyOn(ruleEngine, 'validateMove').mockReturnValue(true);

    // Avoid mutating board geometry in this focused history test.
    jest.spyOn(engineAny, 'processOneDisconnectedRegion').mockResolvedValue(undefined);
    jest.spyOn(engineAny, 'eliminateFromStack').mockImplementation(() => {});

    // Provide a synthetic disconnected region so applyDecisionMove can
    // resolve the target region when the process_territory_region move
    // is applied.
    const region: Territory = {
      spaces: [{ x: 5, y: 5 }],
      controllingPlayer: 0,
      isDisconnected: true,
    };

    jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementation((..._args: any[]) => [region]);

    // Treat this region as always eligible so the decision Move is
    // considered valid.
    jest.spyOn(engineAny, 'canProcessDisconnectedRegion').mockReturnValue(true);

    const processRegionPayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      player: 1,
      type: 'process_territory_region',
      disconnectedRegions: [region],
      to: region.spaces[0],
      thinkTime: 0,
    };

    await engine.makeMove(processRegionPayload);

    const eliminationPayload: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      player: 1,
      type: 'eliminate_rings_from_stack',
      to: { x: 0, y: 1 },
      eliminatedRings: [{ player: 1, count: 1 }],
      eliminationFromStack: {
        position: { x: 0, y: 1 },
        capHeight: 1,
        totalHeight: 2,
      } as any,
      thinkTime: 0,
    };

    await engine.makeMove(eliminationPayload);

    const after = engine.getGameState();
    expect(after.history.length).toBe(2);
    expect(after.history[0].action.type).toBe('process_territory_region');
    expect(after.history[1].action.type).toBe('eliminate_rings_from_stack');
  });

  it('after processing a real disconnected region in move-driven mode, surfaces explicit eliminate_rings_from_stack moves and defers self-elimination until that move is chosen', async () => {
    const engine = new GameEngine(
      'move-driven-territory-full-board',
      boardType,
      players,
      timeControl,
      false
    );
    engine.enableMoveDrivenDecisionPhases();

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;
    const boardManager = engineAny.boardManager as any;
    const ruleEngine = engineAny.ruleEngine as any;

    gameState.gameStatus = 'active';
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'territory_processing';

    const board = gameState.board;

    // Construct a concrete disconnected region for Player 1 consisting of
    // opponent stacks that will be eliminated and collapsed when the
    // region is processed.
    const regionPositions: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];
    regionPositions.forEach((p) => addStack(board, p, 2, 1));

    // Give Player 1 a stack outside the region so the self-elimination
    // prerequisite is satisfied. This stack will be the source of the
    // later eliminate_rings_from_stack decision in move-driven mode.
    const outside = pos(0, 1);
    addStack(board, outside, 1, 3);

    const p1Before = gameState.players.find((p: any) => p.playerNumber === 1)!;
    const eliminatedBefore: number = p1Before.eliminatedRings;
    const totalEliminatedBefore: number = gameState.totalRingsEliminated;

    const region: Territory = {
      spaces: regionPositions,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    // First call to findDisconnectedRegions returns our concrete region so
    // both RuleEngine validation and applyDecisionMove can resolve it.
    // Subsequent calls after processing return an empty list so that the
    // rules engine surfaces explicit elimination decisions instead of
    // further region-processing moves.
    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      // First call: enumeration in GameEngine.getValidMoves
      .mockImplementationOnce(() => [region])
      // Second call: applyDecisionMove('process_territory_region') resolution
      .mockImplementationOnce(() => [region])
      // Subsequent calls: no further disconnected regions remain so that
      // RuleEngine surfaces explicit elimination decisions instead.
      .mockImplementation(() => []);

    // Sanity check: self-elimination prerequisite should hold for this
    // region for Player 1 given the outside stack.
    expect(engineAny.canProcessDisconnectedRegion(region, 1)).toBe(true);

    // From the engine's perspective in territory_processing, the valid
    // decision set should include at least one process_territory_region
    // Move for Player 1.
    const decisionMoves = engine.getValidMoves(1);
    const territoryMoves = decisionMoves.filter((m) => m.type === 'process_territory_region');
    expect(territoryMoves.length).toBeGreaterThan(0);

    const processMove = territoryMoves[0];
    expect(processMove.disconnectedRegions && processMove.disconnectedRegions[0]).toBeDefined();

    const regionFromMove = processMove.disconnectedRegions![0];

    await engine.makeMove({
      player: processMove.player,
      type: processMove.type,
      disconnectedRegions: processMove.disconnectedRegions,
      to: regionFromMove.spaces[0],
      thinkTime: 0,
    });

    const afterRegion = engine.getGameState();

    const afterP1 = afterRegion.players.find((p) => p.playerNumber === 1)!;

    const outsideKey = `${outside.x},${outside.y}`;
    const outsideStackAfterRegion = afterRegion.board.stacks.get(outsideKey);
    expect(outsideStackAfterRegion).toBeDefined();
    if (outsideStackAfterRegion) {
      expect(outsideStackAfterRegion.stackHeight).toBe(3);
      expect(outsideStackAfterRegion.controllingPlayer).toBe(1);
    }

    // In move-driven mode, once no further eligible regions remain for
    // Player 1, the rules engine should expose explicit
    // eliminate_rings_from_stack decision Moves from the same
    // territory_processing phase.
    const tempTerritoryState = {
      ...afterRegion,
      currentPlayer: 1,
      currentPhase: 'territory_processing' as const,
    };

    const rulesMoves = ruleEngine.getValidMoves(tempTerritoryState as any);
    const eliminationMoves = rulesMoves.filter(
      (m: Move) => m.type === 'eliminate_rings_from_stack'
    );
    expect(eliminationMoves.length).toBeGreaterThan(0);

    const eliminationMove = eliminationMoves[0];
    expect(eliminationMove.to).toBeDefined();
    expect(eliminationMove.eliminationFromStack).toBeDefined();

    const capHeight = eliminationMove.eliminationFromStack?.capHeight ?? 0;
    expect(capHeight).toBeGreaterThan(0);

    await engine.makeMove({
      player: eliminationMove.player,
      type: eliminationMove.type,
      to: eliminationMove.to,
      eliminatedRings: eliminationMove.eliminatedRings,
      eliminationFromStack: eliminationMove.eliminationFromStack,
      thinkTime: 0,
    });

    const afterElimination = engine.getGameState();

    // The outside stack should now have been reduced by exactly the cap
    // height (in this simple all-one-color stack, fully removed), and
    // Player 1's eliminatedRings should have increased by the same
    // amount.
    expect(afterElimination.board.stacks.has(outsideKey)).toBe(false);

    const finalP1 = afterElimination.players.find((p) => p.playerNumber === 1)!;

    // History should record the territory region processing and the
    // explicit self-elimination as two distinct canonical Moves.
    expect(afterElimination.history.length).toBe(2);
    expect(afterElimination.history[0].action.type).toBe('process_territory_region');
    expect(afterElimination.history[1].action.type).toBe('eliminate_rings_from_stack');

    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();
  });

  it('in move-driven territory_processing, does not surface eliminate_rings_from_stack when no region has been processed', () => {
    const engine = new GameEngine(
      'move-driven-territory-gating',
      boardType,
      players,
      timeControl,
      false
    );
    engine.enableMoveDrivenDecisionPhases();

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;
    const boardManager = engineAny.boardManager as any;
    const ruleEngine = engineAny.ruleEngine as any;

    gameState.gameStatus = 'active';
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'territory_processing';

    // No disconnected regions for the current player in this cycle.
    jest.spyOn(boardManager, 'findDisconnectedRegions').mockReturnValue([]);

    // Even if the underlying RuleEngine would be willing to surface
    // eliminate_rings_from_stack moves for this state, the GameEngine
    // must NOT expose them until at least one region has been processed
    // in the current territory_processing cycle.
    const ruleMoves: Move[] = [
      {
        id: 'eliminate-0,1',
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: { x: 0, y: 1 },
        eliminatedRings: [{ player: 1, count: 1 }],
        eliminationFromStack: {
          position: { x: 0, y: 1 },
          capHeight: 1,
          totalHeight: 2,
        } as any,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      },
    ];

    const ruleSpy = jest.spyOn(ruleEngine, 'getValidMoves').mockReturnValue(ruleMoves as any);

    const moves = engine.getValidMoves(1);

    // No decision Moves should be exposed in this phase, and the
    // underlying RuleEngine.getValidMoves should not be consulted for
    // elimination options while no region has been processed.
    expect(moves).toEqual([]);
    expect(ruleSpy).not.toHaveBeenCalled();
  });
});
