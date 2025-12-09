import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  BoardType,
  GameState,
  Move,
  Position,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import {
  positionToString,
  enumeratePlacementPositions,
  applyPlacementMoveAggregate,
  evaluateSkipPlacementEligibilityAggregate,
} from '../../src/shared/engine';

/**
 * Sandbox placement parity tests against the shared TS placement core.
 *
 * These tests ensure that:
 * - enumerateLegalRingPlacements (sandbox) is a thin adapter over
 *   enumeratePlacementPositions (shared aggregate),
 * - tryPlaceRings / applyCanonicalMove produce the same board + hand
 *   effects as applyPlacementMoveAggregate when no lines/territory fire,
 * - skip_placement visibility matches the canonical aggregate semantics
 *   plus the backend tightening around ringsInHand > 0.
 */
describe('ClientSandboxEngine placement parity vs shared core', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // For these tests we never actually surface PlayerChoices, but provide
      // a trivial handler to satisfy the constructor.
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as any as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('enumerateLegalRingPlacements matches enumeratePlacementPositions (including NDP)', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    state.currentPhase = 'ring_placement';

    const board = state.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Corner no-dead-placement scenario mirroring the backend shared test:
    // placing at (0,0) with all outward rays immediately blocked must be
    // illegal under canonical NDP semantics.
    const corner: Position = { x: 0, y: 0 };
    const blocked: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    blocked.forEach((p) => {
      board.collapsedSpaces.set(positionToString(p), 0);
    });

    const canonical = enumeratePlacementPositions(state, 1)
      .map((p) => positionToString(p))
      .sort();
    const sandbox = (engineAny.enumerateLegalRingPlacements(1) as Position[])
      .map((p: Position) => positionToString(p))
      .sort();

    expect(sandbox).toEqual(canonical);
    expect(canonical).not.toContain(positionToString(corner));
  });

  test('applyPlacementMoveAggregate matches sandbox tryPlaceRings on empty board', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const initialState: GameState = engine.getGameState();

    const move: Move = {
      id: '',
      type: 'place_ring',
      player: 1,
      from: undefined,
      to: { x: 3, y: 3 } as Position,
      placementCount: 1,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    // Canonical aggregate application from the same logical starting state.
    const coreOutcome = applyPlacementMoveAggregate(initialState, move);
    const coreState = coreOutcome.nextState;

    // Drive the same placement through the sandbox helper.
    // Reset engine internal state to the same starting snapshot.
    engineAny.gameState = initialState as GameState;

    const placed = await engine.tryPlaceRings(move.to as Position, move.placementCount ?? 1);
    expect(placed).toBe(true);

    const sandboxState = engine.getGameState();

    // Compare per-player ringsInHand.
    const coreRings = new Map<number, number>();
    coreState.players.forEach((p) => coreRings.set(p.playerNumber, p.ringsInHand));

    sandboxState.players.forEach((p) => {
      expect(p.ringsInHand).toBe(coreRings.get(p.playerNumber));
    });

    // Compare board stacks: for every occupied cell, stackHeight + ring owners
    // must match between sandbox and core.
    const stackToKeyedSummary = (s: GameState) => {
      const summary: Record<
        string,
        { controllingPlayer: number; stackHeight: number; rings: number[] }
      > = {};
      for (const [key, stack] of s.board.stacks.entries()) {
        summary[key] = {
          controllingPlayer: stack.controllingPlayer,
          stackHeight: stack.stackHeight,
          rings: [...stack.rings],
        };
      }
      return summary;
    };

    const coreStacks = stackToKeyedSummary(coreState);
    const sandboxStacks = stackToKeyedSummary(sandboxState);
    expect(sandboxStacks).toEqual(coreStacks);
  });

  test('skip_placement visibility matches aggregate eligibility + ringsInHand tightening', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    state.currentPhase = 'ring_placement';

    const board = state.board;
    board.stacks.clear();
    board.collapsedSpaces.clear();
    board.markers.clear();

    const stackPos: Position = { x: 3, y: 3 };
    const stackKey = positionToString(stackPos);
    board.stacks.set(stackKey, {
      position: stackPos,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    });

    // Case 1: ringsInHand > 0 and aggregate says skip is eligible → sandbox
    // should surface skip_placement alongside place_ring.
    state.players = state.players.map((p) => (p.playerNumber === 1 ? { ...p, ringsInHand: 3 } : p));

    let eligibility = evaluateSkipPlacementEligibilityAggregate(state, 1);
    const aggregateEligible =
      (eligibility as any).eligible ?? (eligibility as any).canSkip ?? false;
    expect(aggregateEligible).toBe(true);

    let moves = engine.getValidMoves(1);
    expect(moves.some((m) => m.type === 'place_ring')).toBe(true);
    expect(moves.some((m) => m.type === 'skip_placement')).toBe(true);

    // Case 2: aggregate now correctly rejects skip when ringsInHand == 0
    // (per canonical rules, use no_placement_action instead).
    state.players = state.players.map((p) => (p.playerNumber === 1 ? { ...p, ringsInHand: 0 } : p));

    eligibility = evaluateSkipPlacementEligibilityAggregate(state, 1);
    const aggregateRejectsWhenNoRings =
      (eligibility as any).eligible ?? (eligibility as any).canSkip ?? false;
    expect(aggregateRejectsWhenNoRings).toBe(false);
    expect((eligibility as any).code).toBe('NO_RINGS_IN_HAND');

    moves = engine.getValidMoves(1);
    expect(moves.some((m) => m.type === 'skip_placement')).toBe(false);
  });
});
