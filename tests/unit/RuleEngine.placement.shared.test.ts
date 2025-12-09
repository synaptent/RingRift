import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { GameEngine } from '../../src/server/game/GameEngine';
import type { BoardType, GameState, Move, Position } from '../../src/shared/types/game';
import { evaluateSkipPlacementEligibilityAggregate } from '../../src/shared/engine';
import { createTestGameState, pos, addStack, addCollapsedSpace } from '../utils/fixtures';

describe('RuleEngine placement semantics – shared engine parity', () => {
  function createPlacementState(boardType: BoardType): {
    gameState: GameState;
    boardManager: BoardManager;
    ruleEngine: RuleEngine;
  } {
    const gameState = createTestGameState({ boardType });

    const boardManager = new BoardManager(boardType);
    gameState.board = boardManager.createBoard();

    const ruleEngine = new RuleEngine(boardManager, boardType as any);

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'ring_placement';

    return { gameState, boardManager, ruleEngine };
  }

  it('rejects placements that leave no legal move or capture (no-dead-placement)', () => {
    // Mirrors the sandbox no‑dead‑placement scenario from
    // ClientSandboxEngine.placementForcedElimination: placing in the
    // corner with all outward rays immediately blocked must be illegal.
    const { gameState, boardManager, ruleEngine } = createPlacementState('square8');

    const corner = pos(0, 0);
    const blockPositions: Position[] = [pos(1, 0), pos(0, 1), pos(1, 1)];
    for (const p of blockPositions) {
      addCollapsedSpace(gameState.board, p, 0);
    }

    const move: Move = {
      id: 'place-dead',
      type: 'place_ring',
      player: 1,
      from: undefined,
      to: corner,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    expect(ruleEngine.validateMove(move, gameState)).toBe(false);

    const validMoves = ruleEngine.getValidMoves(gameState);
    expect(
      validMoves.some(
        (m) => m.type === 'place_ring' && m.to && m.to.x === corner.x && m.to.y === corner.y
      )
    ).toBe(false);
  });

  it('does not offer skip_placement when player has rings but controls no stacks', () => {
    const { gameState, boardManager, ruleEngine } = createPlacementState('square8');

    // Ensure player 1 has rings in hand and no stacks on board.
    gameState.players = gameState.players.map((p) =>
      p.playerNumber === 1 ? { ...p, ringsInHand: 3 } : p
    );

    const moves = ruleEngine.getValidMoves(gameState);
    expect(moves.some((m) => m.type === 'place_ring')).toBe(true);
    expect(moves.some((m) => m.type === 'skip_placement')).toBe(false);
  });

  it('offers skip_placement only when placement is optional (ringsInHand>0 and legal actions exist)', () => {
    const { gameState, boardManager, ruleEngine } = createPlacementState('square8');

    // Ensure player 1 has rings in hand and at least one controlled stack
    // with a legal movement or capture, so placement is optional.
    gameState.players = gameState.players.map((p) =>
      p.playerNumber === 1 ? { ...p, ringsInHand: 3 } : p
    );

    addStack(gameState.board, pos(3, 3), 1, 1);

    const aggregateEligibility = evaluateSkipPlacementEligibilityAggregate(gameState, 1);
    expect((aggregateEligibility as any).eligible).toBe(true);

    const moves = ruleEngine.getValidMoves(gameState);
    expect(moves.some((m) => m.type === 'place_ring')).toBe(true);
    expect(moves.some((m) => m.type === 'skip_placement')).toBe(true);
  });

  it('rejects skip_placement when ringsInHand == 0 (aggregate correctly rejects as well)', () => {
    const { gameState, boardManager, ruleEngine } = createPlacementState('square8');

    // Player 1 controls a stack with legal actions but has no rings in hand.
    gameState.players = gameState.players.map((p) =>
      p.playerNumber === 1 ? { ...p, ringsInHand: 0 } : p
    );

    addStack(gameState.board, pos(3, 3), 1, 1);

    // The aggregate correctly rejects skip_placement when player has no rings in hand.
    // Per the rules, skip_placement is only valid when rings exist to potentially place;
    // if ringsInHand == 0, the player should use no_placement_action instead.
    const aggregateEligibility = evaluateSkipPlacementEligibilityAggregate(gameState, 1);
    expect(aggregateEligibility.eligible).toBe(false);
    expect(aggregateEligibility.code).toBe('NO_RINGS_IN_HAND');

    const moves = ruleEngine.getValidMoves(gameState);
    expect(moves.some((m) => m.type === 'skip_placement')).toBe(false);

    const skipMove: Move = {
      id: 'skip-test',
      type: 'skip_placement',
      player: 1,
      from: undefined,
      // Sentinel coordinate; semantics are phase-only.
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    expect(ruleEngine.validateMove(skipMove, gameState)).toBe(false);
  });

  it('applies forced elimination when no player has placements, movements, or captures', () => {
    const boardType: BoardType = 'square8';
    const gameState = createTestGameState({ boardType });
    const boardManager = new BoardManager(boardType);
    gameState.board = boardManager.createBoard();

    // Two-player scenario where BOTH players:
    // - have no rings in hand (no placements),
    // - each control exactly one blocked stack with no legal movement, and
    // - have no legal captures between them (paths blocked by collapsed spaces).
    //
    // This matches the global forced-elimination precondition used by the
    // backend test helper resolveBlockedStateForCurrentPlayerForTesting:
    // hasValidPlacements == false, hasValidMovements == false,
    // hasValidCaptures == false for every player with material.
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';
    gameState.players = gameState.players.map((p) => ({
      ...p,
      ringsInHand: 0,
    }));

    // Player 1 blocked in the (0,0) corner.
    const cornerP1 = pos(0, 0);
    addStack(gameState.board, cornerP1, 1, 2);
    addCollapsedSpace(gameState.board, pos(1, 0), 0);
    addCollapsedSpace(gameState.board, pos(0, 1), 0);
    addCollapsedSpace(gameState.board, pos(1, 1), 0);

    // Player 2 blocked in the opposite corner so that no capture path exists.
    const cornerP2 = pos(7, 7);
    addStack(gameState.board, cornerP2, 2, 2);
    addCollapsedSpace(gameState.board, pos(6, 7), 0);
    addCollapsedSpace(gameState.board, pos(7, 6), 0);
    addCollapsedSpace(gameState.board, pos(6, 6), 0);

    // Create a GameEngine instance and inject our prepared GameState so we
    // can drive the test-only resolver that uses the canonical placement /
    // movement / capture availability checks (hasValidPlacements based on
    // enumeratePlacementPositions + no-dead-placement).
    const engine = new GameEngine(
      gameState.id,
      boardType,
      gameState.players as any,
      gameState.timeControl as any,
      false
    );
    const engineAny = engine as any;
    engineAny.gameState = gameState;

    const before = engine.getGameState();
    const beforeStacks = before.board.stacks.size;
    const beforeElimsP1 = before.players.find((p) => p.playerNumber === 1)!.eliminatedRings;

    engineAny.resolveBlockedStateForCurrentPlayerForTesting();

    const after = engine.getGameState();
    const afterStacks = after.board.stacks.size;
    const afterElimsP1 = after.players.find((p) => p.playerNumber === 1)!.eliminatedRings;

    // With no legal placement, movement, or capture available for ANY player
    // with material, the resolver must apply a forced elimination step using
    // the same canonical availability checks as TurnEngine
    // (RR‑CANON‑R072/R100). We assert that:
    // - Player 1 has strictly more eliminated rings than before, and
    // - The total number of stacks on the board has not increased.
    expect(afterElimsP1).toBeGreaterThan(beforeElimsP1);
    expect(afterStacks).toBeLessThanOrEqual(beforeStacks);
  });
});
