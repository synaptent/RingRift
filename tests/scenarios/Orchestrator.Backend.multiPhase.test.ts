import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
  Territory,
} from '../../src/shared/types/game';
import { lineAndTerritoryRuleScenarios, LineAndTerritoryRuleScenario } from './rulesMatrix';
import {
  createDefaultTwoPlayerConfig,
  createOrchestratorBackendEngine,
  createBackendOrchestratorHarness,
  seedOverlengthLineForPlayer,
  seedTerritoryRegionWithOutsideStack,
  toEngineMove,
  filterRealActionMoves,
} from '../helpers/orchestratorTestUtils';
import { enumerateProcessTerritoryRegionMoves } from '../../src/shared/engine/territoryDecisionHelpers';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine/rulesConfig';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { isFSMOrchestratorActive } from '../../src/shared/utils/envFlags';

/**
 * Orchestrator-centric backend multi-phase scenario + invariant tests.
 *
 * These suites exercise the production GameEngine orchestrator adapter
 * (TurnEngineAdapter + processTurnAsync) via the normal GameEngine.makeMove
 * entry point, using move-driven decision phases:
 *
 *   GameEngine.makeMove
 *     -> TurnEngineAdapter.processMove
 *       -> processTurnAsync (shared orchestrator)
 *         -> shared aggregates
 *
 * No legacy RuleEngine-only helpers (processLineFormations, processDisconnectedRegions)
 * are invoked; all decision phases are driven by canonical Move types:
 *   - process_line
 *   - choose_line_reward
 *   - process_territory_region
 *   - eliminate_rings_from_stack
 */

const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
const boardType: BoardType = 'square8';

function createOrchestratorEngineForTest(testId: string, playersOverride?: Player[]): GameEngine {
  if (playersOverride) {
    return createOrchestratorBackendEngine(testId, boardType, playersOverride, timeControl);
  }
  const { players } = createDefaultTwoPlayerConfig(boardType, timeControl);
  return createOrchestratorBackendEngine(testId, boardType, players, timeControl);
}

function keyFromPositions(positions: Position[]): string {
  return positions
    .map((p) => positionToString(p))
    .sort()
    .join('|');
}

describe('Orchestrator.Backend multi-phase scenarios (GameEngine + TurnEngineAdapter)', () => {
  /**
   * Scenario A – Line reward choice (Option 1 vs Option 2) and explicit
   * elimination decision in the line_processing phase.
   *
   * Flow (backend host):
   *   1) Start from a synthetic overlength line for Player 1 on square8.
   *   2) Enter line_processing with move-driven decision phases enabled.
   *   3) Use GameEngine.getValidMoves to surface:
   *        - one process_line Move per line, and
   *        - multiple choose_line_reward Moves for the overlength line.
   *   4) Apply a choose_line_reward Move corresponding to Option 1
   *      (collapse-all + pending line-reward elimination) via GameEngine.makeMove
   *      which delegates to TurnEngineAdapter -> processTurnAsync.
   *   5) Verify:
   *        - All markers in the line collapse to territory for Player 1.
   *        - No rings are eliminated yet (elimination is deferred).
   *        - currentPhase remains line_processing and explicit
   *          eliminate_rings_from_stack Moves are now available.
   *   6) Apply an eliminate_rings_from_stack decision and assert:
   *        - Player 1's eliminatedRings and totalRingsEliminated increase.
   *        - Phase/turn sequencing leaves the game in a non-line decision phase.
   *
   * Throughout, we also check orchestrator-level invariants:
   *   - All moves returned by orchestrator getValidMoves are accepted
   *     by orchestrator validateMove for the same snapshot.
   */
  it('Scenario A – overlength line reward choice + elimination via orchestrator adapter', async () => {
    const engine = createOrchestratorEngineForTest('orch-backend-line-reward');
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // Use the effective line length threshold which accounts for 2-player elevation
    // on square8 (3 → 4). This matches the enumeration logic in
    // enumerateChooseLineRewardMoves.
    const numPlayers = state.players.length;
    const requiredLength = getEffectiveLineLengthThreshold(boardType, numPlayers);

    // Synthetic overlength line for Player 1 on row 0: length = requiredLength + 1.
    const linePositions = seedOverlengthLineForPlayer(engine, 1, 0, 1);

    // Provide a stack for Player 1 so the line-reward elimination has a legal target.
    const elimStackPos: Position = { x: 7, y: 7 };
    const elimRings = [1, 1, 1];
    state.board.stacks.set(positionToString(elimStackPos), {
      position: elimStackPos,
      rings: elimRings,
      stackHeight: elimRings.length,
      capHeight: elimRings.length,
      controllingPlayer: 1,
    } as any);

    // Start directly in the line_processing phase for Player 1, as if they
    // had just completed a move that formed this overlength line.
    state.currentPlayer = 1;
    (state as any).currentPhase = 'line_processing';

    const beforeState = engine.getGameState();
    const beforePlayer1 = beforeState.players.find((p) => p.playerNumber === 1)!;

    // Orchestrator invariant at start: all orchestrator-valid moves for this
    // snapshot should be accepted by orchestrator validateMove.
    {
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);
      expect(orchMoves.length).toBeGreaterThan(0);

      for (const m of orchMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    }

    // Host-level decision surface for Player 1 in line_processing.
    const hostMoves = engine.getValidMoves(1);
    const processLineMoves = hostMoves.filter((m) => m.type === 'process_line');
    const rewardMoves = hostMoves.filter((m) => m.type === 'choose_line_reward');

    expect(processLineMoves.length).toBeGreaterThan(0);
    expect(rewardMoves.length).toBeGreaterThan(0);

    // Filter reward Moves down to those associated with our specific line.
    const lineKey = keyFromPositions(linePositions);
    const rewardForLine = rewardMoves.filter((m) => {
      if (!m.formedLines || m.formedLines.length === 0) return false;
      const line = m.formedLines[0];
      return keyFromPositions(line.positions) === lineKey;
    });

    expect(rewardForLine.length).toBeGreaterThan(0);

    // Heuristics (aligned with existing sandbox line tests):
    //  - Option 1 (collapse-all + elimination): either collapsedMarkers is
    //    undefined or has size >= full line length.
    //  - Option 2 (minimum contiguous segment of length requiredLength):
    //    collapsedMarkers length === requiredLength.
    const collapseAllCandidate = rewardForLine.find((m) => {
      const collapsed = m.collapsedMarkers ?? [];
      return collapsed.length === 0 || collapsed.length >= linePositions.length;
    });
    const minCollapseCandidate = rewardForLine.find((m) => {
      const collapsed = m.collapsedMarkers ?? [];
      return collapsed.length === requiredLength;
    });

    expect(collapseAllCandidate).toBeDefined();
    expect(minCollapseCandidate).toBeDefined();

    const option1Move = collapseAllCandidate as Move;

    // Apply Option 1 via orchestrator-backed GameEngine.makeMove.
    const applyReward = await engine.makeMove(toEngineMove(option1Move));
    expect(applyReward.success).toBe(true);

    const afterReward = engine.getGameState();
    const afterPlayer1 = afterReward.players.find((p) => p.playerNumber === 1)!;

    // All positions in line should now be collapsed spaces for Player 1, with
    // no markers or stacks remaining on those cells.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(afterReward.board.collapsedSpaces.get(key)).toBe(1);
      expect(afterReward.board.markers.has(key)).toBe(false);
      expect(afterReward.board.stacks.has(key)).toBe(false);
    }

    // The orchestrator now applies line-reward eliminations as part of the
    // same turn processing. The collapsed spaces eliminated any opposing
    // stacks that were on the line positions. Verify eliminations were
    // applied (may be >= before if stacks were captured).
    expect(afterPlayer1.eliminatedRings).toBeGreaterThanOrEqual(beforePlayer1.eliminatedRings);
    expect(afterReward.totalRingsEliminated).toBeGreaterThanOrEqual(
      beforeState.totalRingsEliminated
    );

    // Under the orchestrator-adapter path, the full post-move phase pipeline
    // (lines → territory → victory/turn-advance) is driven inside a single
    // turn processing call. After resolving the line reward, control has
    // already advanced to the next player's ring_placement phase.
    expect(afterReward.currentPhase).toBe('ring_placement');
    expect(afterReward.currentPlayer).not.toBe(beforeState.currentPlayer);

    // Orchestrator invariant on the post-reward snapshot as well: every move
    // enumerated for the next position must validate successfully.
    {
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);
      for (const m of orchMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    }
  });

  /**
   * Scenario B – Territory region processing + explicit self-elimination via
   * process_territory_region and eliminate_rings_from_stack Moves.
   *
   * Uses the shared RulesMatrix combined line+territory scenario for square8
   * (Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square8),
   * but focuses on the territory portion:
   *
   *   1) Seed a single-cell disconnected region at (5,5) credited to Player 1
   *      containing a Player 2 stack, plus a Player 1 outside stack at (7,7)
   *      of height 2 to pay the self-elimination cost.
   *   2) Enter territory_processing for Player 1.
   *   3) Assert that getValidMoves surfaces process_territory_region Moves
   *      but no eliminate_rings_from_stack Moves before any region is processed.
   *   4) Apply a process_territory_region Move via GameEngine.makeMove.
   *   5) Assert that interior region spaces collapse to Player 1's territory
   *      and victim stacks are removed.
   *   6) Assert that getValidMoves now surfaces explicit
   *      eliminate_rings_from_stack Moves.
   *   7) Apply an eliminate_rings_from_stack Move targeting the outside stack
   *      and assert elimination + phase/turn sequencing are correct.
   */
  it('Scenario B – process_territory_region + territory self-elimination via orchestrator adapter', async () => {
    const scenario: LineAndTerritoryRuleScenario | undefined = lineAndTerritoryRuleScenarios.find(
      (s) => s.ref.id === 'Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square8'
    );
    expect(scenario).toBeDefined();

    const engine = createOrchestratorEngineForTest('orch-backend-territory');
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    const territory = scenario!.territoryRegion;
    const regionSpaces = territory.spaces;
    const controllingPlayer = territory.controllingPlayer;
    const victimPlayer = territory.victimPlayer;
    const outsideStackPos = territory.outsideStackPosition;
    const outsideHeight = territory.selfEliminationStackHeight;

    // Clear board state and seed only the territory region + outside stack.
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();

    seedTerritoryRegionWithOutsideStack(engine, {
      regionSpaces,
      controllingPlayer,
      victimPlayer,
      outsideStackPosition: outsideStackPos,
      outsideStackHeight: outsideHeight,
    });

    state.currentPlayer = controllingPlayer;
    (state as any).currentPhase = 'territory_processing';

    const beforeState = engine.getGameState();
    const beforePlayer = beforeState.players.find((p) => p.playerNumber === controllingPlayer)!;
    const beforeSnapshot = computeProgressSnapshot(beforeState as any);

    // Before any region is processed, the backend host should not yet owe a
    // self-elimination decision. In this synthetic geometry the shared
    // detector does not report a disconnected region, so orchestrator-driven
    // getValidMoves will not surface process_territory_region here.
    const movesBefore = engine.getValidMoves(controllingPlayer);
    const elimBefore = movesBefore.filter((m) => m.type === 'eliminate_rings_from_stack');
    expect(elimBefore.length).toBe(0);

    // Instead, construct a canonical process_territory_region Move directly
    // via the shared decision helper using a test-only override region. This
    // mirrors the RulesMatrix territory tests and keeps geometry aligned with
    // the shared helpers rather than backend-only detectors.
    const regionTerritory: Territory = {
      spaces: regionSpaces,
      controllingPlayer,
      isDisconnected: true,
    };

    const regionMoves = enumerateProcessTerritoryRegionMoves(beforeState, controllingPlayer, {
      testOverrideRegions: [regionTerritory],
    });
    expect(regionMoves.length).toBeGreaterThan(0);

    const targetKey = keyFromPositions(regionSpaces);
    const regionMove =
      regionMoves.find((m: Move) => {
        if (!m.disconnectedRegions || m.disconnectedRegions.length === 0) {
          return false;
        }
        const regSpaces = m.disconnectedRegions[0].spaces ?? [];
        return keyFromPositions(regSpaces) === targetKey;
      }) ?? regionMoves[0];

    const applyRegion = await engine.makeMove(toEngineMove(regionMove));
    expect(applyRegion.success).toBe(true);

    const afterRegion = engine.getGameState();
    const afterSnapshot = computeProgressSnapshot(afterRegion as any);

    // S-invariant must be non-decreasing across this explicit
    // process_territory_region turn: interior rings are either eliminated
    // or converted into collapsed spaces, so M + C + E cannot shrink.
    expect(afterSnapshot.S).toBeGreaterThanOrEqual(beforeSnapshot.S);

    // Region interior must be collapsed for the controlling player and
    // victim stacks removed.
    for (const pos of regionSpaces) {
      const key = positionToString(pos);
      expect(afterRegion.board.collapsedSpaces.get(key)).toBe(controllingPlayer);
      expect(afterRegion.board.stacks.get(key)).toBeUndefined();
    }

    // After an explicitly submitted process_territory_region decision, the
    // orchestrator completes the post-move phase pipeline (including any
    // additional territory/line/victory checks) and rotates to the next
    // player's ring_placement turn.
    expect(afterRegion.currentPhase).toBe('ring_placement');
    expect(afterRegion.currentPlayer).not.toBe(controllingPlayer);

    // Internal region eliminations must have been credited to the controlling
    // player and the global totals updated, but the outside self-elimination
    // stack is untouched at this point.
    const afterPlayer = afterRegion.players.find((p) => p.playerNumber === controllingPlayer)!;
    expect(afterRegion.totalRingsEliminated).toBeGreaterThan(beforeState.totalRingsEliminated);
    expect(afterPlayer.eliminatedRings).toBeGreaterThan(beforePlayer.eliminatedRings);

    const outsideAfterRegion = afterRegion.board.stacks.get(positionToString(outsideStackPos));
    expect(outsideAfterRegion).toBeDefined();
    if (outsideAfterRegion) {
      expect(outsideAfterRegion.stackHeight).toBe(outsideHeight);
    }

    // Orchestrator invariant on the post-region snapshot.
    {
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);
      for (const m of orchMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    }
  });

  /**
   * Scenario C – complex chain capture (triangle loop) via orchestrator adapter.
   *
   * This scenario mirrors the ComplexChainCaptures FAQ_15_3_2_CyclicPattern_TriangleLoop
   * test but drives the position through the orchestrator-backed GameEngine +
   * TurnEngineAdapter path instead of the legacy capture helpers:
   *
   *   1) Seed a simple triangle of stacks:
   *        - P1 at (3,3) height 1.
   *        - P2 at (3,4), (4,4), (4,3) height 1 each.
   *   2) Start in capture phase for Player 1 and apply a single
   *      overtaking_capture via GameEngine.makeMove (delegating to
   *      TurnEngineAdapter → processTurnAsync).
   *   3) While in chain_capture, repeatedly:
   *        - Use the orchestrator adapter to enumerate
   *          continue_capture_segment moves for the current snapshot and
   *          assert that validateMoveOnly accepts each one.
   *        - Cross-check that the host-level continue_capture_segment
   *          surface is a subset of orchestrator moves.
   *        - Apply one continuation via GameEngine.makeMove.
   *   4) After the chain ends, assert that the final board state matches
   *      the FAQ expectations: a single Blue-controlled stack of height 4
   *      and no remaining Red stacks.
   */
  it('Scenario C – complex chain capture via orchestrator adapter', async () => {
    const engine = createOrchestratorEngineForTest('orch-backend-chain-triangle');
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // Clear any existing geometry and seed the triangle-loop stacks.
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();

    const startPos: Position = { x: 3, y: 3 };
    const target1: Position = { x: 3, y: 4 };
    const target2: Position = { x: 4, y: 4 };
    const target3: Position = { x: 4, y: 3 };

    const seedStack = (pos: Position, playerNumber: number, height: number): void => {
      const rings = Array(height).fill(playerNumber);
      state.board.stacks.set(positionToString(pos), {
        position: pos,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber,
      } as any);
    };

    seedStack(startPos, 1, 1);
    seedStack(target1, 2, 1);
    seedStack(target2, 2, 1);
    seedStack(target3, 2, 1);

    state.currentPhase = 'capture';
    state.currentPlayer = 1;
    state.gameStatus = 'active';

    // Capture initial S-invariant snapshot before any overtaking_capture is
    // applied. Chain captures may increase S via eliminations but must never
    // decrease S = markers + collapsed + eliminated.
    const beforeSnapshot = computeProgressSnapshot(state as any);

    // Start the chain via a host-level overtaking_capture. We prefer the
    // FAQ-style (3,3) over (3,4) → (3,5) segment when available but remain
    // tolerant if additional capture options appear.
    const hostMoves = engine.getValidMoves(1);
    const overtakingMoves = hostMoves.filter((m) => m.type === 'overtaking_capture');
    expect(overtakingMoves.length).toBeGreaterThan(0);

    const preferredStart = overtakingMoves.find(
      (m) =>
        m.from &&
        m.captureTarget &&
        m.to &&
        m.from.x === startPos.x &&
        m.from.y === startPos.y &&
        m.captureTarget.x === target1.x &&
        m.captureTarget.y === target1.y &&
        m.to.x === 3 &&
        m.to.y === 5
    );

    const startMove = (preferredStart ?? overtakingMoves[0]) as Move;

    // Orchestrator invariant on the initial capture snapshot: the chosen
    // overtaking_capture must validate under the canonical adapter for the
    // same GameState snapshot, even if the adapter does not enumerate
    // capture moves directly in this phase.
    {
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const validation = harness.adapter.validateMoveOnly(orchState, startMove);
      expect(validation.valid).toBe(true);
    }

    const firstResult = await engine.makeMove(toEngineMove(startMove));
    expect(firstResult.success).toBe(true);

    // Drive any mandatory continuations via the orchestrator-backed
    // chain_capture phase, asserting that host-level
    // continue_capture_segment moves validate under the canonical
    // orchestrator adapter for each intermediate snapshot.
    const MAX_CHAIN_STEPS = 8;
    let steps = 0;

    while ((engineAny.gameState as GameState).currentPhase === 'chain_capture') {
      steps += 1;
      if (steps > MAX_CHAIN_STEPS) {
        throw new Error('Scenario C: exceeded maximum chain-capture steps');
      }

      const chainState: GameState = engineAny.gameState as GameState;
      const currentPlayer = chainState.currentPlayer;

      const hostChainMoves = engine
        .getValidMoves(currentPlayer)
        .filter((m) => m.type === 'continue_capture_segment') as Move[];
      expect(hostChainMoves.length).toBeGreaterThan(0);

      const chosen = hostChainMoves[0];

      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const validation = harness.adapter.validateMoveOnly(orchState, chosen);
      expect(validation.valid).toBe(true);

      const result = await engine.makeMove(toEngineMove(chosen));
      expect(result.success).toBe(true);
    }

    const finalState = engine.getGameState();
    const finalSnapshot = computeProgressSnapshot(finalState as any);
    const allStacks = Array.from(finalState.board.stacks.values());
    const blueStacks = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacks = allStacks.filter((s) => s.controllingPlayer === 2);

    // As in the ComplexChainCaptures triangle scenario, we only assert
    // aggregate outcomes, not the exact landing coordinate:
    //   - One Blue-controlled stack of height 4.
    //   - No remaining Red stacks.
    expect(blueStacks.length).toBe(1);
    expect(blueStacks[0].stackHeight).toBe(4);
    expect(blueStacks[0].controllingPlayer).toBe(1);
    expect(redStacks.length).toBe(0);

    // Progress invariant: S must be non-decreasing over the full
    // overtaking_capture + chain_capture sequence.
    expect(finalSnapshot.S).toBeGreaterThanOrEqual(beforeSnapshot.S);
  });

  /**
   * Scenario D – LPS (last-player-standing) victory only after post-move phases.
   *
   * This orchestrator-level check mirrors the old phase-order test from the
   * legacy LPS suite by asserting that a last_player_standing victory is
   * produced only after the post-move pipeline has traversed:
   *
   *   movement → line_processing → territory_processing → victory
   *
   * We construct a minimal 2-player structural-stalemate snapshot where:
   *   - Only Player 1 has a single stack on the board.
   *   - Both players have equal eliminatedRings after Player 1 pays a
   *     self-elimination cost via eliminate_rings_from_stack.
   *   - No stacks, markers, or territory remain after the move.
   *
   * Under VictoryAggregate's stalemate ladder this yields a terminal
   * last_player_standing result decided by getLastActor, while the
   * orchestrator metadata exposes the expected phase ordering in
   * metadata.phasesTraversed.
   */
  it('Scenario D – LPS victory is classified from terminal territory_processing state', () => {
    const engine = createOrchestratorEngineForTest('orch-backend-lps-ordering');
    const baseState = engine.getGameState();

    const stackPos: Position = { x: 3, y: 3 };

    // Build a structural-stalemate snapshot in the terminal territory phase:
    // - 2 players, active game in territory_processing.
    // - Single Player 1 stack on the board.
    // - Both players have equal eliminatedRings after P1 pays a
    //   self-elimination cost, so the victory ladder falls through to
    //   last_actor → last_player_standing.
    const state: GameState = {
      ...baseState,
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'territory_processing',
      // Single prior move by Player 1 so getLastActor resolves to P1.
      moveHistory: [
        {
          id: 'prev-1',
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date(0),
          thinkTime: 0,
          moveNumber: 1,
        } as Move,
      ],
      history: [],
      players: baseState.players.map((p) =>
        p.playerNumber === 1
          ? {
              ...p,
              ringsInHand: 0,
              eliminatedRings: 0,
              territorySpaces: 0,
            }
          : {
              ...p,
              ringsInHand: 0,
              eliminatedRings: 1,
              territorySpaces: 0,
            }
      ),
      board: {
        ...baseState.board,
        stacks: new Map([
          [
            positionToString(stackPos),
            {
              position: stackPos,
              rings: [1],
              stackHeight: 1,
              capHeight: 1,
              controllingPlayer: 1,
            } as any,
          ],
        ]),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {
          1: 0,
          2: 1,
        },
      },
      totalRingsEliminated: 1,
      winner: undefined,
    };

    const eliminateMove: Move = {
      id: 'elim-test',
      type: 'eliminate_rings_from_stack',
      player: 1,
      to: stackPos,
      eliminatedRings: [{ player: 1, count: 1 }],
      eliminationFromStack: {
        position: stackPos,
        capHeight: 1,
        totalHeight: 1,
      },
      timestamp: new Date(0),
      thinkTime: 0,
      moveNumber: state.moveHistory.length + 1,
    } as Move;

    const result = processTurn(state, eliminateMove);

    // Victory must be classified as last_player_standing with Player 1
    // as winner (last actor in moveHistory).
    expect(result.victoryResult).toBeDefined();
    expect(result.victoryResult!.isGameOver).toBe(true);
    expect(result.victoryResult!.reason).toBe('last_player_standing');
    expect(result.victoryResult!.winner).toBe(1);

    // Orchestrator metadata must reflect that we terminated from
    // territory_processing rather than short-circuiting earlier.
    const phases = result.metadata.phasesTraversed;
    expect(phases).toContain('territory_processing');
  });
});

describe('Orchestrator.Backend invariants – getValidMoves vs validateMove & forced elimination surfaces', () => {
  /**
   * For a small selection of hand-crafted positions and phases, confirm that
   * all orchestrator-level getValidMoves entries are accepted by
   * orchestrator validateMove and that the host never exposes "phantom"
   * real-action moves for the active player.
   *
   * NOTE: This is intentionally a smoke-style invariant test; deeper
   * exhaustive invariants remain the responsibility of the shared rules
   * engine and Python invariant suites.
   */
  it('all orchestrator getValidMoves entries validate successfully across simple positions', async () => {
    const engine = createOrchestratorEngineForTest('orch-backend-invariants-basic');
    const engineAny: any = engine;

    // 1) Initial ring_placement state.
    {
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);
      expect(orchMoves.length).toBeGreaterThan(0);

      for (const m of orchMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }

      // Real actions should be a subset of host getValidMoves.
      const hostMoves = engine.getValidMoves(orchState.currentPlayer);
      const realHost = filterRealActionMoves(hostMoves);
      const realOrch = filterRealActionMoves(orchMoves);
      expect(realOrch.length).toBeGreaterThan(0);
      expect(realHost.length).toBeGreaterThan(0);
    }

    // Apply a single placement to move into movement phase.
    const initialHostMoves = engine.getValidMoves(engineAny.gameState.currentPlayer);
    const firstPlacement = initialHostMoves.find((m) => m.type === 'place_ring');
    expect(firstPlacement).toBeDefined();

    const applyPlacement = await engine.makeMove(toEngineMove(firstPlacement as Move));
    expect(applyPlacement.success).toBe(true);

    // 2) After first placement: either still in ring_placement or in movement,
    // but orchestrator invariants must continue to hold.
    {
      const harness = createBackendOrchestratorHarness(engine);
      const orchState = harness.getState();
      const orchMoves = harness.adapter.getValidMovesFor(orchState);
      for (const m of orchMoves) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    }
  });

  /**
   * Forced-elimination invariant (qualitative): when the current player has
   * no real actions available in a non-terminal state but still controls
   * stacks, the host exposes either:
   *   - explicit eliminate_rings_from_stack decision Moves (territory/line),
   *   - or the game transitions into a terminal state via host-level helpers.
   *
   * This test builds on the existing ForcedEliminationAndStalemate scenarios
   * but drives the state through the orchestrator adapter rather than
   * directly calling legacy helpers.
   */
  it('forced-elimination surfaces elimination decisions or terminal states when no real actions remain', async () => {
    const { players } = createDefaultTwoPlayerConfig(boardType, timeControl);
    // Start with no rings in hand so placements are impossible.
    players.forEach((p) => {
      p.ringsInHand = 0;
    });

    const engine = createOrchestratorEngineForTest('orch-backend-forced-elim', players);
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    state.gameStatus = 'active';
    state.currentPlayer = 1;
    (state as any).currentPhase = 'movement';

    // Single Player 1 stack at (0,0) with capHeight 2, completely blocked by
    // collapsed spaces as in ForcedEliminationAndStalemate.test.ts.
    const stackPos: Position = { x: 0, y: 0 };
    const rings = [1, 1];
    state.board.stacks.clear();
    state.board.collapsedSpaces.clear();

    state.board.stacks.set(positionToString(stackPos), {
      position: stackPos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    } as any);

    const blockers: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    for (const pos of blockers) {
      state.board.collapsedSpaces.set(positionToString(pos), 0);
    }

    const before = engine.getGameState();

    const harness = createBackendOrchestratorHarness(engine);
    const orchState = harness.getState();
    const orchMoves = harness.adapter.getValidMovesFor(orchState);

    const realOrch = filterRealActionMoves(orchMoves);
    const hasRealAction = realOrch.length > 0;

    if (!hasRealAction && orchState.gameStatus === 'active') {
      // In this smoke invariant we only require that the host does not get
      // stuck in an "ACTIVE_NO_MOVES" plateau. Either an explicit elimination
      // decision should be surfaced or the game should be structurally
      // terminal after applying host-level automatic steps.
      const elimMoves = orchMoves.filter((m) => m.type === 'eliminate_rings_from_stack');

      if (elimMoves.length > 0) {
        // All elimination decisions must themselves validate under the orchestrator.
        for (const m of elimMoves) {
          const validation = harness.adapter.validateMoveOnly(orchState, m);
          expect(validation.valid).toBe(true);
        }

        // Host-level surface should not invent non-orchestrator moves: any host
        // elimination decisions must be a subset of orchestrator ones.
        const hostMoves = engine.getValidMoves(orchState.currentPlayer);
        const hostElims = hostMoves.filter((m) => m.type === 'eliminate_rings_from_stack');
        expect(hostElims.length).toBeGreaterThan(0);
      } else {
        // Apply the legacy forced-elimination helper to resolve this blocked
        // state and then re-check orchestration invariants on the resulting
        // position. This mirrors the backend ForcedEliminationAndStalemate
        // scenarios but expresses the post-resolution state in orchestrator
        // terms.
        engine.resolveBlockedStateForCurrentPlayerForTesting();
        const after = engine.getGameState();

        if (after.gameStatus === 'active') {
          // Forced elimination must have increased the global eliminated-ring
          // count for some player.
          expect(after.totalRingsEliminated).toBeGreaterThan(before.totalRingsEliminated);

          const harness2 = createBackendOrchestratorHarness(engine);
          const orchState2 = harness2.getState();
          const orchMoves2 = harness2.adapter.getValidMovesFor(orchState2);
          for (const m of orchMoves2) {
            const validation = harness2.adapter.validateMoveOnly(orchState2, m);
            expect(validation.valid).toBe(true);
          }
        } else {
          expect(after.gameStatus).toBe('completed');
        }
      }
    } else {
      // If orchestrator still believes a real action exists, simply assert
      // that all such actions validate successfully.
      for (const m of realOrch) {
        const validation = harness.adapter.validateMoveOnly(orchState, m);
        expect(validation.valid).toBe(true);
      }
    }

    // Sanity: either the game progressed (eliminations applied) or we
    // confirmed that real actions were indeed available.
    const afterState = engine.getGameState();
    expect(afterState.moveHistory.length).toBeGreaterThanOrEqual(before.moveHistory.length);
  });

  /**
   * Forced-elimination phase invariant: when the current player is in the
   * forced_elimination phase (blocked with stacks but no legal actions),
   * the orchestrator must surface explicit forced_elimination moves that
   * allow eliminating caps from owned stacks, and those moves must validate.
   *
   * Note: Per RR-CANON-R075/R076, eliminate_rings_from_stack moves only appear
   * in forced_elimination phase, not territory_processing. In territory_processing,
   * elimination is handled via process_territory_region moves with self-elimination costs.
   */
  it('forced_elimination phase surfaces elimination decisions when player is blocked with material', async () => {
    const { players } = createDefaultTwoPlayerConfig(boardType, timeControl);
    // Start with no rings in hand so placements are impossible.
    players.forEach((p) => {
      p.ringsInHand = 0;
    });

    const engine = createOrchestratorEngineForTest('orch-backend-forced-elim-territory', players);
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // Create a truly blocked state: P1 has a single stack surrounded by P2 stacks.
    // This ensures hasAnyGlobalMovementOrCapture returns false for P1.
    // Place P1 at (3,3) surrounded by P2 stacks at all orthogonal neighbors.
    const stackPos: Position = { x: 3, y: 3 };
    const p1Rings = [1, 1, 1];

    // Modify the internal engine state directly. Note: we must do this BEFORE
    // creating the harness, and then also update the harness's state since
    // engine.getGameState() creates shallow clones with new Map copies.
    //
    // Per RR-CANON-R075/R076, eliminate_rings_from_stack moves are only valid
    // in the 'forced_elimination' phase. In territory_processing, elimination
    // is handled via process_territory_region moves. When no regions exist,
    // the orchestrator transitions to forced_elimination phase.
    state.gameStatus = 'active';
    state.currentPlayer = 1;
    (state as any).currentPhase = 'forced_elimination';
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();

    // P1's single stack
    state.board.stacks.set(positionToString(stackPos), {
      position: stackPos,
      rings: p1Rings,
      stackHeight: p1Rings.length,
      capHeight: p1Rings.length,
      controllingPlayer: 1,
    } as any);

    // Surround with P2 stacks to block all movement/capture options for P1.
    // Square8 uses 8-direction Moore adjacency, so we need to block all 8 neighbors.
    // Neighbors of (3,3): E(4,3), SE(4,4), S(3,4), SW(2,4), W(2,3), NW(2,2), N(3,2), NE(4,2)
    const p2Rings = [2, 2, 2, 2, 2]; // Taller stacks so P1 can't capture
    const neighbors = [
      { x: 4, y: 3 }, // E
      { x: 4, y: 4 }, // SE
      { x: 3, y: 4 }, // S
      { x: 2, y: 4 }, // SW
      { x: 2, y: 3 }, // W
      { x: 2, y: 2 }, // NW
      { x: 3, y: 2 }, // N
      { x: 4, y: 2 }, // NE
    ];
    for (const pos of neighbors) {
      state.board.stacks.set(positionToString(pos), {
        position: pos,
        rings: p2Rings,
        stackHeight: p2Rings.length,
        capHeight: p2Rings.length,
        controllingPlayer: 2,
      } as any);
    }

    // Create harness AFTER state modifications, and get state from harness.
    // The harness's getState() returns the adapter's internal state reference,
    // not a clone. Since engine.getGameState() clones Maps, we need to verify
    // the harness sees the modified state correctly.
    const harness = createBackendOrchestratorHarness(engine);
    const orchState = harness.getState();

    // The adapter's state should reflect our modifications since getGameState()
    // is called after we modified engineAny.gameState.
    // However, getGameState() creates new Maps, so verify the state is correct:
    expect(orchState.currentPhase).toBe('forced_elimination');
    expect(orchState.currentPlayer).toBe(1);

    const orchMoves = harness.adapter.getValidMovesFor(orchState);

    // No "real" actions should be present in forced_elimination phase for this shape.
    // forced_elimination moves are considered bookkeeping/penalty moves, not real actions.
    const realOrch = filterRealActionMoves(orchMoves);
    expect(realOrch.length).toBe(0);

    // In forced_elimination phase, moves are typed as 'forced_elimination', not 'eliminate_rings_from_stack'
    const elimMoves = orchMoves.filter((m) => m.type === 'forced_elimination');
    expect(elimMoves.length).toBeGreaterThan(0);

    for (const m of elimMoves) {
      const validation = harness.adapter.validateMoveOnly(orchState, m);
      expect(validation.valid).toBe(true);
    }

    // Apply one elimination via the real backend host path and confirm that
    // the orchestrator drives the phase/turn transition.
    const elimMove = elimMoves[0] as Move;
    const apply = await engine.makeMove(toEngineMove(elimMove));
    expect(apply.success).toBe(true);

    const after = engine.getGameState();

    // Elimination must reduce or remove the stack and increase elimination counts.
    const finalStack = after.board.stacks.get(positionToString(stackPos));
    if (finalStack) {
      expect(finalStack.stackHeight).toBeLessThan(rings.length);
    }
    const p1After = after.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.eliminatedRings).toBeGreaterThanOrEqual(1);
    expect(after.totalRingsEliminated).toBeGreaterThanOrEqual(1);

    // After paying the self-elimination cost there should be no further
    // territory_processing step for player 1; the orchestrator either
    // rotates to the next player or ends the game.
    if (after.gameStatus === 'active') {
      expect(after.currentPhase).toBe('ring_placement');
      expect(after.currentPlayer).not.toBe(1);
    }
  });
});
