import { GameEngine } from '../../src/server/game/GameEngine';
import type {
  BoardType,
  GameState,
  Player,
  TimeControl,
  Move,
  Position,
} from '../../src/shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../../src/shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine/rulesConfig';
import type { TurnEngineAdapter } from '../../src/server/game/turn/TurnEngineAdapter';
import {
  createSimpleAdapter,
  createAutoSelectDecisionHandler,
} from '../../src/server/game/turn/TurnEngineAdapter';

/**
 * Shared helpers for orchestrator-centric backend and sandbox tests.
 *
 * These utilities intentionally avoid direct access to legacy GameEngine internals
 * like processLineFormations or processDisconnectedRegions and instead build
 * initial GameState geometry that can be exercised via the canonical
 * orchestrator-first host flows (GameEngine.makeMove + TurnEngineAdapter,
 * ClientSandboxEngine.applyCanonicalMove + SandboxOrchestratorAdapter).
 */
export interface BackendOrchestratorHarness {
  engine: GameEngine;
  adapter: TurnEngineAdapter;
  /**
   * Return the current GameState snapshot used by the adapter's state holder.
   * Note: this is independent from GameEngine.getGameState and is intended
   * only for invariants that exercise orchestrator enumeration/validation.
   */
  getState: () => GameState;
}

const DEFAULT_TIME_CONTROL: TimeControl = {
  initialTime: 600,
  increment: 0,
  type: 'blitz',
};

/**
 * Create a simple two-player configuration for the given board type.
 */
export function createDefaultTwoPlayerConfig(
  boardType: BoardType = 'square8',
  timeControl: TimeControl = DEFAULT_TIME_CONTROL
): { players: Player[]; timeControl: TimeControl } {
  const ringsPerPlayer = BOARD_CONFIGS[boardType].ringsPerPlayer;

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: ringsPerPlayer,
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
      ringsInHand: ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  return { players, timeControl };
}

/**
 * Create a GameEngine configured for orchestrator-first turn processing and
 * move-driven decision phases, mirroring the production backend path.
 */
export function createOrchestratorBackendEngine(
  gameId: string,
  boardType: BoardType = 'square8',
  players?: Player[],
  timeControl: TimeControl = DEFAULT_TIME_CONTROL
): GameEngine {
  const config = players
    ? { players, timeControl }
    : createDefaultTwoPlayerConfig(boardType, timeControl);

  const engine = new GameEngine(gameId, boardType, config.players, config.timeControl, true);

  // Ensure move-driven decision phases + orchestrator adapter are enabled
  // so GameEngine.makeMove delegates to TurnEngineAdapter → processTurnAsync.
  engine.enableMoveDrivenDecisionPhases();
  engine.enableOrchestratorAdapter();

  // In tests we bypass the lobby/GameSession readiness handshake and mark all
  // players as ready before calling startGame(). GameEngine's constructor
  // derives isReady from player.type (only AI players are auto-ready), so we
  // explicitly flip the internal flags here rather than changing production
  // code.
  const engineAny = engine as any;
  if (engineAny.gameState && Array.isArray(engineAny.gameState.players)) {
    engineAny.gameState.players.forEach((p: any) => {
      p.isReady = true;
    });
  }

  const started = engine.startGame();
  if (!started) {
    const debugState: GameState | undefined =
      typeof (engine as any).getGameState === 'function'
        ? (engine as any).getGameState()
        : engineAny.gameState;

    console.error('[orchestratorTestUtils] Failed to start orchestrator-enabled GameEngine', {
      gameId,
      boardType,
      players: debugState?.players?.map((p) => ({
        playerNumber: p.playerNumber,
        type: (p as any).type,
        isReady: (p as any).isReady,
        username: (p as any).username,
      })),
      gameStatus: (debugState as any)?.gameStatus,
    });

    throw new Error('Failed to start orchestrator-enabled GameEngine for test');
  }

  return engine;
}

/**
 * Create a simple TurnEngineAdapter harness for the current GameState of a
 * backend GameEngine. This is primarily used for invariant tests that
 * compare orchestrator getValidMoves vs validateMove.
 */
export function createBackendOrchestratorHarness(engine: GameEngine): BackendOrchestratorHarness {
  const initial = engine.getGameState();
  const { adapter, getState } = createSimpleAdapter(initial, createAutoSelectDecisionHandler());
  return { engine, adapter, getState };
}

/**
 * Helper to normalise a canonical Move emitted by getValidMoves into the
 * Omit<Move, 'id' | 'timestamp' | 'moveNumber'> form expected by
 * GameEngine.makeMove / RulesBackendFacade.applyMove.
 */
export function toEngineMove(move: Move): Omit<Move, 'id' | 'timestamp' | 'moveNumber'> {
  const { id, timestamp, moveNumber, ...rest } = move as any;
  return {
    ...rest,
    thinkTime: 0,
  };
}

/**
 * Seed an overlength horizontal marker line for the given player on a
 * square board. Returns the list of positions that form the line.
 *
 * This helper mirrors the geometry used in the existing line scenario tests
 * but does not stub any detection helpers; orchestrator + shared
 * lineDecisionHelpers are responsible for discovering the line.
 */
export function seedOverlengthLineForPlayer(
  engine: GameEngine,
  playerNumber: number,
  rowIndex = 0,
  overlengthBy = 1
): Position[] {
  const engineAny = engine as any;
  const state: GameState = engineAny.gameState as GameState;
  const board = state.board;
  // Use the effective line length threshold which accounts for 2-player elevation
  // on square8 (3 → 4). This ensures the seeded line is actually overlength
  // relative to the threshold used by enumerateChooseLineRewardMoves.
  const requiredLength = getEffectiveLineLengthThreshold(
    state.boardType,
    state.players.length
  );
  const length = requiredLength + overlengthBy;

  // Clear any pre-existing markers on the target row to avoid interference.
  for (let x = 0; x < board.size; x += 1) {
    const key = positionToString({ x, y: rowIndex } as Position);
    board.markers.delete(key);
    // Leave stacks/collapsedSpaces untouched; callers control them.
  }

  const positions: Position[] = [];
  for (let x = 0; x < length; x += 1) {
    const pos = { x, y: rowIndex } as Position;
    positions.push(pos);
    board.markers.set(positionToString(pos), {
      player: playerNumber,
      position: pos,
      type: 'regular',
    } as any);
  }

  return positions;
}

export interface TerritoryRegionSeed {
  regionSpaces: Position[];
  controllingPlayer: number;
  victimPlayer: number;
  outsideStackPosition: Position;
  outsideStackHeight: number;
}

/**
 * Seed a minimal disconnected-region + outside-stack configuration suitable
 * for territory-processing and self-elimination scenarios.
 *
 * The regionSpaces are populated with victim stacks of height 1; the
 * controlling player receives a single outside stack with the requested
 * height.
 */
export function seedTerritoryRegionWithOutsideStack(
  engine: GameEngine,
  seed: TerritoryRegionSeed
): void {
  const engineAny = engine as any;
  const state: GameState = engineAny.gameState as GameState;
  const board = state.board;

  // Clear any existing stacks at the specified positions.
  for (const pos of seed.regionSpaces) {
    board.stacks.delete(positionToString(pos));
  }
  board.stacks.delete(positionToString(seed.outsideStackPosition));

  // Place victim stacks inside the region.
  for (const pos of seed.regionSpaces) {
    const rings = [seed.victimPlayer];
    board.stacks.set(positionToString(pos), {
      position: pos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: seed.victimPlayer,
    } as any);
  }

  // Outside stack for the controlling player used to pay the
  // self-elimination cost.
  const outsideRings = Array(seed.outsideStackHeight).fill(seed.controllingPlayer);
  board.stacks.set(positionToString(seed.outsideStackPosition), {
    position: seed.outsideStackPosition,
    rings: outsideRings,
    stackHeight: outsideRings.length,
    capHeight: outsideRings.length,
    controllingPlayer: seed.controllingPlayer,
  } as any);
}

/**
 * Utility to filter a Move[] down to "real actions" – placements,
 * movements, and captures – excluding explicit decision moves like
 * process_line, choose_line_reward, process_territory_region, and
 * eliminate_rings_from_stack.
 */
export function filterRealActionMoves(moves: Move[]): Move[] {
  const realTypes: Move['type'][] = [
    'place_ring',
    'skip_placement',
    'move_ring',
    'move_stack',
    'overtaking_capture',
    'continue_capture_segment',
  ];
  return moves.filter((m) => realTypes.includes(m.type));
}
