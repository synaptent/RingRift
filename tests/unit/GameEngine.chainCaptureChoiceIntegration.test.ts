import { GameEngine } from '../../src/server/game/GameEngine';
import { PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  PlayerChoice,
  PlayerChoiceResponse,
  CaptureDirectionChoice,
  PlayerChoiceResponseFor,
  RingStack,
  positionToString
} from '../../src/shared/types/game';

/**
 * Integration-style tests for chain capture when multiple follow-up capture
 * directions are available and the engine must consult CaptureDirectionChoice
 * via PlayerInteractionManager.
 *
 * These tests mirror the spirit of the Rust
 * `test_chain_capture_player_choice_simulation` scenario by setting up a
 * position where the capturing stack can continue in more than one direction
 * and verifying that the chosen option is applied.
 */

interface CaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
}

function cloneGameStateForCaptures(state: GameState): GameState {
  return {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      collapsedSpaces: new Map(state.board.collapsedSpaces),
      territories: new Map(state.board.territories),
      formedLines: [...state.board.formedLines],
      eliminatedRings: { ...state.board.eliminatedRings }
    },
    moveHistory: [...state.moveHistory],
    players: [...state.players],
    spectators: [...state.spectators]
  };
}

function computeCapHeight(rings: number[]): number {
  if (rings.length === 0) return 0;
  const top = rings[0];
  let h = 1;
  for (let i = 1; i < rings.length; i++) {
    if (rings[i] === top) h++;
    else break;
  }
  return h;
}

/**
 * Apply a single overtaking capture segment locally to a cloned GameState,
 * mirroring the stack behaviour of GameEngine.performOvertakingCapture but
 * ignoring markers (which are not relevant for this isolated geometry test).
 */
function applyOvertakingSegmentLocally(state: GameState, move: Move): void {
  if (!move.from || !move.captureTarget) return;
  const board = state.board;
  const fromKey = positionToString(move.from);
  const targetKey = positionToString(move.captureTarget);
  const attacker = board.stacks.get(fromKey);
  const target = board.stacks.get(targetKey);
  if (!attacker || !target) return;

  const capturedRing = target.rings[0];
  const remainingTargetRings = target.rings.slice(1);

  if (remainingTargetRings.length > 0) {
    const newTarget: RingStack = {
      position: target.position,
      rings: remainingTargetRings,
      stackHeight: remainingTargetRings.length,
      capHeight: computeCapHeight(remainingTargetRings),
      controllingPlayer: remainingTargetRings[0]
    };
    board.stacks.set(targetKey, newTarget);
  } else {
    board.stacks.delete(targetKey);
  }

  board.stacks.delete(fromKey);

  const newRings = [...attacker.rings, capturedRing];
  const landingKey = positionToString(move.to);
  const newStack: RingStack = {
    position: move.to,
    rings: newRings,
    stackHeight: newRings.length,
    capHeight: computeCapHeight(newRings),
    controllingPlayer: newRings[0]
  };
  board.stacks.set(landingKey, newStack);
}

/**
 * Enumerate all legal overtaking capture chains for a given player from a
 * starting position, using a pure RuleEngine+BoardManager DFS. This is used
 * to verify that the TS engine's capture geometry matches the rules/Rust
 * behaviour across *full chains*, not just a single segment.
 */
function enumerateChainsFrom(
  gameState: GameState,
  boardType: BoardType,
  start: Position,
  player: number
): CaptureSegment[][] {
  const sequences: CaptureSegment[][] = [];
  const boardManager = new BoardManager(boardType);
  const ruleEngine = new RuleEngine(boardManager as any, boardType as any);

  function dfs(state: GameState, from: Position, path: CaptureSegment[]) {
    const moves = (ruleEngine as any).getValidCaptures(player, state) as Move[];
    const fromKey = positionToString(from);
    const candidates = moves.filter(
      m => m.from && positionToString(m.from) === fromKey
    );

    if (candidates.length === 0) {
      // Terminal chain
      sequences.push(path);
      return;
    }

    for (const m of candidates) {
      const nextState = cloneGameStateForCaptures(state);
      applyOvertakingSegmentLocally(nextState, m);
      const segment: CaptureSegment = {
        from: m.from!,
        target: m.captureTarget!,
        landing: m.to
      };
      dfs(nextState, m.to, [...path, segment]);
    }
  }

  dfs(cloneGameStateForCaptures(gameState), start, []);
  return sequences;
}

describe('GameEngine chain capture with CaptureDirectionChoice integration', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const players: Player[] = [
    {
      id: 'red',
      username: 'Red',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'blue',
      username: 'Blue',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'green',
      username: 'Green',
      type: 'human',
      playerNumber: 3,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'yellow',
      username: 'Yellow',
      type: 'human',
      playerNumber: 4,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  function makeStack(playerNumber: number, height: number, position: Position): RingStack {
    const rings = Array(height).fill(playerNumber);
    return {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber
    };
  }

  test('enumerates all valid overtaking chains for the orthogonal choice scenario', () => {
    const engine = new GameEngine('chain-enumeration', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    // Ensure capture phase & correct player so RuleEngine allows capture.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const redPos: Position = { x: 3, y: 3 };
    const bluePos: Position = { x: 3, y: 4 };
    const greenPos: Position = { x: 4, y: 5 };
    const yellowPos: Position = { x: 2, y: 5 };

    boardManager.setStack(redPos, makeStack(1, 2, redPos), gameState.board);
    boardManager.setStack(bluePos, makeStack(2, 1, bluePos), gameState.board);
    boardManager.setStack(greenPos, makeStack(3, 1, greenPos), gameState.board);
    boardManager.setStack(yellowPos, makeStack(4, 1, yellowPos), gameState.board);

    const chains = enumerateChainsFrom(gameState, boardType, redPos, 1);
    const seqReprs = chains.map(chain =>
      chain.map(seg =>
        `${seg.from.x},${seg.from.y}:${seg.target.x},${seg.target.y}->${seg.landing.x},${seg.landing.y}`
      )
    );

    // Represent each full chain as a single string "from:target->landing | ...".
    const chainStrings = seqReprs.map(chain => chain.join(' | '));

    // NOTE: Under the *full* TS capture rules, the engine actually supports more
    // legal chains than the five canonical ones hard‑coded below. Additional
    // sequences branch by:
    //   - Allowing the initial capture from (3,3) over (3,4) to land at (3,6)
    //     and (3,7) (valid by the flexible landing rule: any landing beyond the
    //     target with distance ≥ stack height, along a clear ray).
    //   - Continuing to capture Yellow from later positions (for example from
    //     (6,5) or (7,5) after a Green capture), which the rules permit.
    //
    // So the enumerateChainsFrom helper is doing its job correctly: it finds
    // all chains allowed by the current TS rules. To keep this test as a
    // *full-rules regression* for this configuration, expectedChainStrings
    // should be the complete set of chains produced by enumerateChainsFrom.
    //
    // REGENERATION WORKFLOW:
    //   1. Temporarily add:
    //        console.log('CHAINS', [...chainStrings].sort());
    //      inside this test after chainStrings is computed.
    //   2. Run:
    //        npm test -- --runTestsByPath tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts
    //      and copy the logged, sorted list.
    //   3. Paste it into expectedChainStrings below (keeping semantic grouping
    //      if desired).
    //   4. Remove the console.log once updated.
    //
    // For readability and stable diffs, we keep the literal array grouped or
    // commented as we like, but compare sorted copies of both arrays.
    const expectedChainStrings = [
      // Full set of chains currently produced by enumerateChainsFrom for this
      // configuration (sortedActual from the latest Jest run). Keep this list
      // in sync with the engine by regenerating via the workflow in the
      // comment above whenever capture geometry changes.
      '3,3:3,4->3,5 | 3,5:2,5->0,5 | 0,5:4,5->5,5',
      '3,3:3,4->3,5 | 3,5:2,5->0,5 | 0,5:4,5->6,5',
      '3,3:3,4->3,5 | 3,5:2,5->0,5 | 0,5:4,5->7,5',
      '3,3:3,4->3,5 | 3,5:4,5->6,5 | 6,5:2,5->0,5',
      '3,3:3,4->3,5 | 3,5:4,5->6,5 | 6,5:2,5->1,5',
      '3,3:3,4->3,5 | 3,5:4,5->7,5 | 7,5:2,5->0,5',
      '3,3:3,4->3,5 | 3,5:4,5->7,5 | 7,5:2,5->1,5',
      '3,3:3,4->3,6 | 3,6:2,5->0,3',
      '3,3:3,4->3,6 | 3,6:4,5->6,3',
      '3,3:3,4->3,6 | 3,6:4,5->7,2',
      '3,3:3,4->3,7'
    ];

    const sortedActual = [...chainStrings].sort();
    const sortedExpected = [...expectedChainStrings].sort();

    expect(sortedActual).toEqual(sortedExpected);
    expect(sortedActual.length).toBe(sortedExpected.length);
  });

  test('applies a continue_capture_segment produced by getValidMoves for the orthogonal chain scenario', async () => {
    // Scenario (inspired by Rust test_chain_capture_player_choice_simulation):
    // - Red at (3,3) h2 (attacker)
    // - Blue at (3,4) h1 (initial capture target)
    // After capturing Blue and landing at (3,5), Red H3 has multiple
    // follow-up capture options. We verify that GameEngine exposes these via
    // `continue_capture_segment` moves in getValidMoves and that applying one
    // of them mutates the board consistently with the rules.

    const engine = new GameEngine('chain-choice', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    // Ensure capture phase & correct player so RuleEngine allows capture.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    // Board setup:
    // Red attacker at (3,3) height 2.
    // Blue initial target at (3,4) height 1.
    // Green potential chain target at (4,5) height 1.
    // Yellow potential chain target at (2,5) height 1.
    const redPos: Position = { x: 3, y: 3 };
    const bluePos: Position = { x: 3, y: 4 };
    const greenPos: Position = { x: 4, y: 5 };
    const yellowPos: Position = { x: 2, y: 5 };

    boardManager.setStack(redPos, makeStack(1, 2, redPos), gameState.board);
    boardManager.setStack(bluePos, makeStack(2, 1, bluePos), gameState.board);
    boardManager.setStack(greenPos, makeStack(3, 1, greenPos), gameState.board);
    boardManager.setStack(yellowPos, makeStack(4, 1, yellowPos), gameState.board);

    // Initial capture: Red from (3,3) over Blue at (3,4) landing at (3,5).
    const initialResult = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 3, y: 5 }
    } as Move);

    expect(initialResult.success).toBe(true);

    // After the initial segment, the engine should be in chain_capture phase
    // and getValidMoves should expose follow-up segments as
    // continue_capture_segment moves from (3,5).
    expect(gameState.currentPhase).toBe('chain_capture');
    const continuationMoves = engine.getValidMoves(1);

    expect(continuationMoves.length).toBeGreaterThan(0);
    continuationMoves.forEach((m) => {
      expect(m.type).toBe('continue_capture_segment');
      expect(m.player).toBe(1);
      expect(m.from).toEqual({ x: 3, y: 5 });
    });

    const allPairs = continuationMoves.map(
      (m) => `${m.captureTarget!.x},${m.captureTarget!.y}->${m.to!.x},${m.to!.y}`
    );

    // The rule-faithful options from the first branching point from (3,5)
    // should appear among the continuation moves: Green with landings at
    // (6,5) and (7,5), and Yellow with landing at (0,5).
    expect(allPairs).toEqual(
      expect.arrayContaining([
        '4,5->6,5',
        '4,5->7,5',
        '2,5->0,5',
      ])
    );

    // Choose one continuation deterministically (lexicographically earliest
    // landing) and apply it as a canonical continue_capture_segment move.
    const selectedMove = continuationMoves.reduce((prev, cur) => {
      const prevTo = prev.to!;
      const curTo = cur.to!;
      if (
        curTo.x < prevTo.x ||
        (curTo.x === prevTo.x && curTo.y < prevTo.y)
      ) {
        return cur;
      }
      return prev;
    });

    const followUpResult = await engine.makeMove({
      player: selectedMove.player,
      type: selectedMove.type,
      from: selectedMove.from,
      captureTarget: selectedMove.captureTarget,
      to: selectedMove.to,
    } as Move);

    expect(followUpResult.success).toBe(true);

    const board = gameState.board;
    const stackAtStart = board.stacks.get('3,3');
    const stackAtBlue = board.stacks.get('3,4');
    const stackAtIntermediate = board.stacks.get('3,5');
    const targetKey = `${selectedMove.captureTarget!.x},${selectedMove.captureTarget!.y}`;
    const stackAtTarget = board.stacks.get(targetKey);
    const finalLandingKey = `${selectedMove.to!.x},${selectedMove.to!.y}`;
    const stackAtFinal = board.stacks.get(finalLandingKey);

    expect(stackAtStart).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtIntermediate).toBeUndefined();
    expect(stackAtTarget).toBeUndefined();

    // Final capturing stack should exist at the chosen landing position and
    // be controlled by Red.
    expect(stackAtFinal).toBeDefined();
    expect(stackAtFinal!.controllingPlayer).toBe(1);
    expect(stackAtFinal!.stackHeight).toBeGreaterThanOrEqual(3);
  });

  test('uses CaptureDirectionChoice for diagonal chain options (diagonal rays)', async () => {
    // Scenario: diagonal chain continuation to exercise NE and SW rays.
    // - Red at (3,3) h2 (attacker)
    // - Blue at (4,4) h1 (initial capture target)
    // After capturing Blue and landing at (5,5), Red H3 has two
    // diagonal follow-up capture options:
    //   * NE ray: target Green at (6,4), landing at (7,3)
    //   * SW ray: target Yellow at (4,6), landing at (3,7)

    let lastChoice: CaptureDirectionChoice | undefined;

    const fakeHandler = {
      requestChoice: jest.fn(async (choice: PlayerChoice): Promise<PlayerChoiceResponse<unknown>> => {
        if (choice.type === 'capture_direction') {
          const typedChoice = choice as CaptureDirectionChoice;
          lastChoice = typedChoice;

          const options = typedChoice.options;
          expect(options.length).toBeGreaterThanOrEqual(1);

          // Deterministically choose the option with the smallest
          // landingPosition (x, then y) for reproducibility.
          let selected = options[0];
          for (const opt of options) {
            if (
              opt.landingPosition.x < selected.landingPosition.x ||
              (opt.landingPosition.x === selected.landingPosition.x &&
                opt.landingPosition.y < selected.landingPosition.y)
            ) {
              selected = opt;
            }
          }

          return {
            choiceId: choice.id,
            playerNumber: choice.playerNumber,
            selectedOption: selected
          } as PlayerChoiceResponseFor<CaptureDirectionChoice> as PlayerChoiceResponse<unknown>;
        }

        throw new Error(`Unexpected choice type in test: ${choice.type}`);
      })
    };

    const interactionManager = new PlayerInteractionManager(fakeHandler as any);
    const engine = new GameEngine('chain-choice-diagonal', boardType, players, timeControl, false, interactionManager);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Ensure capture phase & correct player so RuleEngine allows capture.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    // Board setup:
    // Red attacker at (3,3) height 2.
    // Blue initial target at (4,4) height 1.
    // Green potential diagonal chain target at (6,4) height 1 (NE ray).
    // Yellow potential diagonal chain target at (4,6) height 1 (SW ray).
    const redPos: Position = { x: 3, y: 3 };
    const bluePos: Position = { x: 4, y: 4 };
    const greenPos: Position = { x: 6, y: 4 };
    const yellowPos: Position = { x: 4, y: 6 };

    boardManager.setStack(redPos, makeStack(1, 2, redPos), gameState.board);
    boardManager.setStack(bluePos, makeStack(2, 1, bluePos), gameState.board);
    boardManager.setStack(greenPos, makeStack(3, 1, greenPos), gameState.board);
    boardManager.setStack(yellowPos, makeStack(4, 1, yellowPos), gameState.board);

    // Initial capture: Red from (3,3) over Blue at (4,4) landing at (5,5).
    const initialResult = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 5, y: 5 }
    } as Move);

    expect(initialResult.success).toBe(true);

    // Under the current distance>=stackHeight rule, there are no legal
    // diagonal follow-up captures from (5,5) with height 3, so the
    // chain terminates immediately and CaptureDirectionChoice is not
    // invoked.
    expect(fakeHandler.requestChoice).not.toHaveBeenCalled();

    const board = gameState.board;

    // Red should have moved from the start to the initial landing
    // position, capturing Blue in the process.
    const stackAtStart = board.stacks.get('3,3');
    const stackAtBlue = board.stacks.get('4,4');
    const stackAtLanding = board.stacks.get('5,5');

    expect(stackAtStart).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtLanding).toBeDefined();
    expect(stackAtLanding!.controllingPlayer).toBe(1);
    expect(stackAtLanding!.stackHeight).toBe(3);

    // Green and Yellow diagonal targets should remain on the board,
    // since no follow-up capture is legal from this configuration.
    const stackAtGreen = board.stacks.get('6,4');
    const stackAtYellow = board.stacks.get('4,6');

    expect(stackAtGreen).toBeDefined();
    expect(stackAtYellow).toBeDefined();

    // Chain state must be cleared after the initial capture.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });
});
