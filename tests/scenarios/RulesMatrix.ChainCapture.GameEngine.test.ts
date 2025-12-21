import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  TimeControl,
  RingStack,
  Position,
} from '../../src/shared/types/game';
import { chainCaptureRuleScenarios, ChainCaptureRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → GameEngine chain-capture scenarios
 *
 * This suite exercises chain-capture patterns from FAQ 15.3.1 (180° reversal)
 * and FAQ 15.3.2 (cyclic patterns) using the shared `chainCaptureRuleScenarios`
 * definitions.
 *
 * It is intentionally parallel to tests/scenarios/ComplexChainCaptures.test.ts
 * but parameterised by the shared rulesMatrix so that future chain patterns
 * can be expressed as data.
 */

describe('RulesMatrix → GameEngine chain-capture scenarios (Section 10.3; FAQ 15.3.1–15.3.2)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
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

  function createEngine(boardType: BoardType): {
    engine: GameEngine;
    gameState: GameState;
    boardManager: any;
  } {
    const engine = new GameEngine('rules-matrix-chain', boardType, basePlayers, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: any = engineAny.boardManager;
    return { engine, gameState, boardManager };
  }

  function setupStacks(engine: GameEngine, stacks: ChainCaptureRuleScenario['stacks']): GameState {
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: any = engineAny.boardManager;

    gameState.board.stacks.clear();

    const boardType = gameState.board.type as BoardType;

    for (const s of stacks) {
      const rings = Array(s.height).fill(s.player);

      let position: Position = s.position as Position;
      // For hexagonal boards, ensure we use full cube coordinates even if the
      // rulesMatrix scenario omits z; derive z so that x + y + z = 0.
      if (boardType === 'hexagonal' && (position as any).z == null) {
        position = {
          ...position,
          z: -position.x - position.y,
        };
      }

      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: s.player,
      };
      boardManager.setStack(position, stack, gameState.board);
    }

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'capture';

    return gameState;
  }

  /**
   * Resolve any active capture chain for the current player by repeatedly
   * selecting a continue_capture_segment move from GameEngine.getValidMoves
   * while the game remains in the 'chain_capture' phase.
   *
   * This mirrors the previous engine-driven chain-capture loop but drives it
   * explicitly through the unified Move model.
   */
  async function resolveChainIfPresent(engine: GameEngine): Promise<void> {
    const engineAny: any = engine;

    const MAX_STEPS = 16;
    let steps = 0;

    const initialState: GameState = engineAny.gameState as GameState;
    if (initialState.currentPhase !== 'chain_capture') {
      return;
    }

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      steps++;
      if (steps > MAX_STEPS) {
        throw new Error('resolveChainIfPresent: exceeded maximum chain-capture steps');
      }

      const state: GameState = engineAny.gameState as GameState;
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      // eslint-disable-next-line no-console
      console.log('resolveChainIfPresent (RulesMatrix) debug', {
        phase: state.currentPhase,
        currentPlayer,
        moveCount: moves.length,
        moveTypes: moves.map((m) => m.type),
      });

      const chainMoves = moves.filter((m) => m.type === 'continue_capture_segment');

      expect(chainMoves.length).toBeGreaterThan(0);

      // For these FAQ-driven scenarios, either there is a single mandatory
      // continuation or any choice yields the same aggregate outcome, so we
      // deterministically pick the first option.
      const next = chainMoves[0];

      const result = await engine.makeMove({
        player: next.player,
        type: 'continue_capture_segment',
        from: next.from,
        captureTarget: next.captureTarget,
        to: next.to,
      } as any);

      expect(result.success).toBe(true);
    }
  }

  const scenarios: ChainCaptureRuleScenario[] = chainCaptureRuleScenarios.filter((s) =>
    [
      'Rules_10_3_Q15_3_1_180_degree_reversal_basic',
      'Rules_10_3_Q15_3_2_cyclic_pattern_triangle_loop',
      'Rules_10_3_Q15_3_x_hex_cyclic_triangle_pattern',
      'Rules_10_3_strategic_chain_ending_choice_square8',
      'Rules_10_3_multi_directional_zigzag_chain_square8',
    ].includes(s.ref.id)
  );

  test.each<ChainCaptureRuleScenario>(scenarios)(
    '%s → backend GameEngine chain capture matches FAQ aggregate effects',
    async (scenario) => {
      const { engine } = createEngine(scenario.boardType);
      const gameState = setupStacks(engine, scenario.stacks);

      const board = gameState.board;

      // Execute the first scripted overtaking_capture segment from the
      // scenario. Any mandatory follow-up segments are then resolved via
      // explicit continue_capture_segment moves in the dedicated
      // 'chain_capture' phase.
      const [first] = scenario.moves;

      if (scenario.ref.id === 'Rules_10_3_strategic_chain_ending_choice_square8') {
        // eslint-disable-next-line no-console
        console.log('chain-ending-choice initial', {
          stackKeys: Array.from(board.stacks.keys()),
          stackHeights: Array.from(board.stacks.values()).map((s) => ({
            pos: `${s.position.x},${s.position.y}`,
            height: s.stackHeight,
            controller: s.controllingPlayer,
            rings: s.rings,
          })),
        });
      }

      const firstResult = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: first.from,
        captureTarget: first.captureTarget,
        to: first.to,
      } as any);
      expect(firstResult.success).toBe(true);

      // Resolve any mandatory chain-capture continuation for the active
      // player. When no chain is active, this is a no-op.
      await resolveChainIfPresent(engine);

      const stacks = board.stacks as Map<string, RingStack>;
      const allStacks: RingStack[] = Array.from(stacks.values());

      if (scenario.ref.id === 'Rules_10_3_Q15_3_1_180_degree_reversal_basic') {
        // 180° reversal (FAQ 15.3.1): starting from Blue H4 at A and Red H3 at B,
        // after a legal reversal sequence Blue has H6 and Red has H1 at B.
        const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
        expect(blueStacks.length).toBe(1);

        const finalBlue = blueStacks[0];
        expect(finalBlue.stackHeight).toBe(6);
        expect(finalBlue.controllingPlayer).toBe(1);

        const redAtB = stacks.get('6,4');
        expect(redAtB).toBeDefined();
        expect(redAtB!.stackHeight).toBe(1);
      } else if (scenario.ref.id === 'Rules_10_3_Q15_3_2_cyclic_pattern_triangle_loop') {
        // Cyclic triangle pattern (FAQ 15.3.2): Blue overtakes three Red stacks
        // in a closed loop. We assert only aggregate outcomes, not the exact
        // landing coordinate, to remain tolerant of different but still-legal
        // chain paths chosen by the engine.
        const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
        expect(blueStacks.length).toBe(1);

        const finalBlue = blueStacks[0];
        expect(finalBlue.stackHeight).toBe(4);
        expect(finalBlue.controllingPlayer).toBe(1);

        const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);
        expect(redStacks.length).toBe(0);
      } else if (scenario.ref.id === 'Rules_10_3_Q15_3_x_hex_cyclic_triangle_pattern') {
        // Hexagonal cyclic triangle pattern (FAQ 15.3.x): overtaker moves around
        // an inner triangle of three height-2 targets on a hex board.
        const overtakerStacks = allStacks.filter((s) => s.controllingPlayer === 1);
        const targetStacks = allStacks.filter((s) => s.controllingPlayer === 2);

        expect(overtakerStacks.length).toBe(1);

        const overtakerFinal = overtakerStacks[0];

        // Started from height 2; chain must increase overtaker height.
        expect(overtakerFinal.stackHeight).toBeGreaterThan(2);

        const totalRingsInitial = scenario.stacks.reduce((sum, st) => sum + st.height, 0);
        const totalRingsAfter = allStacks.reduce((sum, st) => sum + st.stackHeight, 0);
        expect(totalRingsAfter).toBe(totalRingsInitial);

        expect(overtakerFinal.controllingPlayer).toBe(1);
        expect(overtakerFinal.rings.some((r) => r === 1)).toBe(true);
        expect(overtakerFinal.rings.some((r) => r !== 1)).toBe(true);

        const otherPlayerStacks = allStacks.filter(
          (s) => s.controllingPlayer !== 1 && s.controllingPlayer !== 2
        );
        expect(otherPlayerStacks.length).toBe(0);

        // Internal chain state should be cleared once no further legal captures exist.
        expect((engine as any).chainCaptureState).toBeUndefined();

        const totalTargetInitial = scenario.stacks
          .filter((st) => st.player === 2)
          .reduce((sum, st) => sum + st.height, 0);
        const totalTargetRings = targetStacks.reduce((sum, st) => sum + st.stackHeight, 0);
        expect(totalTargetRings).toBeLessThan(totalTargetInitial);
      } else if (scenario.ref.id === 'Rules_10_3_strategic_chain_ending_choice_square8') {
        // Strategic chain-ending choice: Player 1 may choose a capture that
        // leads to no further legal captures, even if alternative captures
        // would allow the chain to continue.
        const finalStack = stacks.get('3,5');
        // DEBUG: capture outcome snapshot for server engine divergence

        const engineAny = engine as any;
        const currentState = engineAny.gameState as GameState;
        const lastMove = currentState.moveHistory[currentState.moveHistory.length - 1];
        console.log('chain-ending-choice snapshot', {
          phase: currentState.currentPhase,
          currentPlayer: currentState.currentPlayer,
          lastMove,
          finalStack,
          stackKeys: Array.from(stacks.keys()),
          stackHeights: Array.from(stacks.values()).map((s) => ({
            pos: `${s.position.x},${s.position.y}`,
            height: s.stackHeight,
            controller: s.controllingPlayer,
            rings: s.rings,
          })),
        });
        expect(finalStack).toBeDefined();
        expect(finalStack!.stackHeight).toBe(2);
        expect(finalStack!.controllingPlayer).toBe(1);

        // Other potential targets along the alternative path should remain.
        expect(stacks.get('4,3')).toBeDefined();
        expect(stacks.get('6,3')).toBeDefined();
      } else if (scenario.ref.id === 'Rules_10_3_multi_directional_zigzag_chain_square8') {
        // Multi-directional zig-zag chain: starting from a single overtaking
        // capture, mandatory continuations may change direction between
        // segments while still preserving straight-line geometry per hop.
        // Aggregate expectations mirror ComplexChainCaptures.Multi_Directional_ZigZag_Chain.
        const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
        const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);

        expect(blueStacks.length).toBe(1);
        expect(blueStacks[0].stackHeight).toBe(4);
        expect(blueStacks[0].controllingPlayer).toBe(1);
        expect(redStacks.length).toBe(0);
      } else {
        throw new Error(`Unhandled ChainCaptureRuleScenario id: ${scenario.ref.id}`);
      }
    }
  );
});
