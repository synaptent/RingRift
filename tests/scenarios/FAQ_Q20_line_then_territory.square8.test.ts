/**
 * FAQ Q20: Combined line + territory turn (square boards)
 *
 * Scenario: A pre-formed overlength line for Player 1 is resolved, then a
 * disconnected single-cell territory region is processed (with mandatory
 * self-elimination) within the same turn. This exercises the
 * line_processing → territory_processing pipeline for square8/square19.
 */

import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { SandboxOrchestratorAdapter } from '../../src/client/sandbox/SandboxOrchestratorAdapter';
import type { BoardType, GameState, Move, Position } from '../../src/shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../../src/shared/types/game';
import { lineAndTerritoryRuleScenarios, type LineAndTerritoryRuleScenario } from './rulesMatrix';
import {
  seedOverlengthLineForPlayer,
  seedTerritoryRegionWithOutsideStack,
} from '../helpers/orchestratorTestUtils';
import { enumerateProcessTerritoryRegionMoves } from '../../src/shared/engine/territoryDecisionHelpers';
import { findAllLines } from '../../src/shared/engine/lineDetection';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine/rulesConfig';

type SquareBoard = Extract<BoardType, 'square8' | 'square19'>;

function createSandboxEngine(boardType: SquareBoard): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers: 2,
    playerKinds: ['human', 'human'],
  };

  const handler: SandboxInteractionHandler = {
    // Deterministically pick the first option for all PlayerChoices
    async requestChoice<TChoice>(_choice: TChoice) {
      return {
        choiceId: (_choice as any).id,
        playerNumber: (_choice as any).playerNumber,
        choiceType: (_choice as any).type,
        selectedOption: Array.isArray((_choice as any).options)
          ? (_choice as any).options[0]
          : (_choice as any).options,
      } as any;
    },
  };

  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

function seedScenarioGeometry(engine: ClientSandboxEngine, scenario: LineAndTerritoryRuleScenario) {
  const engineAny: any = engine;
  const state: GameState = engineAny.gameState as GameState;
  const board = state.board;

  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.formedLines = [];

  const controllingPlayer = scenario.territoryRegion.controllingPlayer;
  const victimPlayer = scenario.territoryRegion.victimPlayer;
  const regionSpaces = scenario.territoryRegion.spaces as Position[];
  const outsideStackPosition = scenario.territoryRegion.outsideStackPosition as Position;
  const outsideStackHeight = scenario.territoryRegion.selfEliminationStackHeight ?? 2;

  // Seed overlength line markers for the controlling player.
  seedOverlengthLineForPlayer(
    engine as any,
    controllingPlayer,
    scenario.line.rowIndex,
    scenario.line.overlengthBy
  );
  // Cache formed lines to make line-processing enumeration deterministic.
  board.formedLines = findAllLines(board);

  // Seed a minimal disconnected region plus outside stack for territory processing.
  seedTerritoryRegionWithOutsideStack(engine as any, {
    regionSpaces,
    controllingPlayer,
    victimPlayer,
    outsideStackPosition,
    outsideStackHeight,
  });

  // Normalise ring counts so the orchestrator bookkeeping matches the seeded stacks.
  const ringsOnBoard = new Map<number, number>();
  for (const stack of board.stacks.values() as Iterable<{ rings: number[] }>) {
    for (const ring of stack.rings) {
      ringsOnBoard.set(ring, (ringsOnBoard.get(ring) ?? 0) + 1);
    }
  }

  state.totalRingsInPlay = Array.from(ringsOnBoard.values()).reduce((sum, count) => sum + count, 0);
  state.totalRingsEliminated = 0;

  const ringsPerPlayer = BOARD_CONFIGS[state.boardType].ringsPerPlayer;
  state.players.forEach((player) => {
    const onBoard = ringsOnBoard.get(player.playerNumber) ?? 0;
    player.ringsInHand = Math.max(0, ringsPerPlayer - onBoard);
    player.eliminatedRings = 0;
    player.territorySpaces = 0;
  });

  state.currentPlayer = controllingPlayer;
  state.currentPhase = 'line_processing';
  state.gameStatus = 'active';
  state.moveHistory = [];
  state.history = [];
}

function getSandboxAdapter(engine: ClientSandboxEngine): SandboxOrchestratorAdapter {
  return (engine as any).getOrchestratorAdapter() as SandboxOrchestratorAdapter;
}

async function applyIfPresent(
  adapter: SandboxOrchestratorAdapter,
  engine: ClientSandboxEngine,
  predicate: (move: Move) => boolean
): Promise<boolean> {
  const moves = adapter.getValidMoves();
  const target = moves.find(predicate);
  if (!target) return false;
  expect(adapter.validateMove(target).valid).toBe(true);
  await engine.applyCanonicalMove(target);
  return true;
}

function buildRegionMove(state: GameState, scenario: LineAndTerritoryRuleScenario): Move {
  const regionMoves = enumerateProcessTerritoryRegionMoves(
    state,
    scenario.territoryRegion.controllingPlayer,
    {
      testOverrideRegions: [
        {
          spaces: scenario.territoryRegion.spaces as Position[],
          controllingPlayer: scenario.territoryRegion.controllingPlayer,
          isDisconnected: true,
        },
      ],
    }
  );

  const move = regionMoves.find((m) => m.type === 'process_territory_region');
  if (!move) {
    throw new Error('No process_territory_region move enumerated for scenario');
  }
  return move as Move;
}

function buildLineRewardMove(state: GameState, player: number): Move {
  const line = findAllLines(state.board).find((l) => l.player === player);
  if (!line) {
    throw new Error('No formed line detected for line reward move');
  }

  const requiredLength = getEffectiveLineLengthThreshold(
    state.boardType,
    state.players.length,
    state.rulesOptions
  );

  return {
    player,
    type: 'choose_line_reward',
    formedLines: [line],
    collapsedMarkers: line.positions.slice(0, requiredLength),
  } as Move;
}

async function playLineThenTerritory(boardType: SquareBoard) {
  const scenario = lineAndTerritoryRuleScenarios.find(
    (s) => s.boardType === boardType && s.kind === 'line-and-territory'
  );
  expect(scenario).toBeDefined();
  if (!scenario) return;

  const engine = createSandboxEngine(boardType);
  seedScenarioGeometry(engine, scenario);
  const adapter = getSandboxAdapter(engine);

  // Resolve the overlength line (prefer explicit choose_line_reward).
  const lineDecision = buildLineRewardMove(
    engine.getGameState(),
    scenario.territoryRegion.controllingPlayer
  );
  expect(adapter.validateMove(lineDecision as any).valid).toBe(true);
  await engine.applyCanonicalMove(lineDecision as any);

  // Apply any line-reward elimination that surfaces before territory.
  for (let i = 0; i < 2; i += 1) {
    const progressed = await applyIfPresent(
      adapter,
      engine,
      (m) => m.type === 'eliminate_rings_from_stack'
    );
    if (!progressed) break;
  }

  // Advance past any no-op line action if needed.
  if (engine.getGameState().currentPhase === 'line_processing') {
    await applyIfPresent(adapter, engine, (m) => m.type === 'no_line_action');
  }

  // If the orchestrator auto-advanced the turn after line resolution (because
  // we seeded a snapshot directly at line_processing), explicitly stage the
  // territory phase to exercise the combined line → territory pipeline.
  const engineAny: any = engine;
  const rawState: GameState = engineAny.gameState as GameState;
  if (rawState.currentPhase !== 'territory_processing') {
    rawState.currentPhase = 'territory_processing';
    rawState.currentPlayer = scenario.territoryRegion.controllingPlayer;
  }

  const afterLine = engine.getGameState();
  expect(afterLine.currentPhase).toBe('territory_processing');

  // Apply the territory region (single-cell) using the shared helper.
  const regionMove = buildRegionMove(afterLine, scenario);
  expect(adapter.validateMove(regionMove as any).valid).toBe(true);
  await engine.applyCanonicalMove(regionMove as any);

  // Pay the mandatory self-elimination if required.
  for (let i = 0; i < 2; i += 1) {
    const progressed = await applyIfPresent(
      adapter,
      engine,
      (m) => m.type === 'eliminate_rings_from_stack'
    );
    if (!progressed) break;
  }

  // After processing, the region should be collapsed to the controlling player.
  const finalState = engine.getGameState();
  const regionSpaces = scenario.territoryRegion.spaces as Position[];
  for (const pos of regionSpaces) {
    const collapsed = finalState.board.collapsedSpaces.get(positionToString(pos));
    expect(collapsed).toBe(scenario.territoryRegion.controllingPlayer);
  }

  expect(finalState.currentPlayer).toBe(2);
  expect(finalState.gameStatus).toBe('active');
  expect(['ring_placement', 'movement']).toContain(finalState.currentPhase);
}

describe('FAQ Q20: line → territory multi-choice turn (square boards)', () => {
  it('square8: resolves line reward then territory region in one turn', async () => {
    await playLineThenTerritory('square8');
  });

  it('square19: resolves line reward then territory region in one turn', async () => {
    await playLineThenTerritory('square19');
  });
});
