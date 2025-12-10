import type { Prisma } from '@prisma/client';
import type { BoardType as PrismaBoardType, GameStatus as PrismaGameStatus } from '@prisma/client';
import type {
  GameState,
  Position,
  TimeControl,
  MarkerInfo,
  RingStack,
  Territory,
} from '../../../shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../../../shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../../shared/engine/rulesConfig';
import { generateGameSeed } from '../../../shared/utils/rng';
import { getDatabaseClient } from '../../database/connection';
import { logger } from '../../utils/logger';
import type { GameEngine } from '../GameEngine';

/**
 * Type for accessing GameEngine internals in test fixtures.
 * This is intentionally loose to allow direct state manipulation for test scenarios.
 */
interface GameEngineInternal {
  gameState: GameState;
}

export type DecisionPhaseScenario =
  | 'line_processing'
  | 'territory_processing'
  | 'chain_capture_choice'
  | 'near_victory_elimination'
  | 'near_victory_territory';

export interface DecisionPhaseFixtureMetadata {
  kind: 'decision_phase_fixture';
  scenario: DecisionPhaseScenario;
  version: 1;
  /**
   * Optional per-game timeout override for E2E testing.
   * When set, GameSession will use this instead of the global config timeout.
   * This allows E2E tests to verify timeout behavior without waiting 30+ seconds.
   */
  shortTimeoutMs?: number;
  /**
   * Optional warning time override (milliseconds before timeout to show warning).
   */
  shortWarningBeforeMs?: number;
}

export interface CreateDecisionPhaseFixtureOptions {
  creatorUserId: string;
  scenario?: DecisionPhaseScenario;
  isRated?: boolean;
  /**
   * Optional short timeout for E2E testing (milliseconds).
   * When set, the decision phase will timeout after this duration.
   * Typical value for tests: 3000-5000ms.
   */
  shortTimeoutMs?: number;
  /**
   * Optional short warning time (milliseconds before timeout).
   * Typical value for tests: 1000-2000ms.
   */
  shortWarningBeforeMs?: number;
}

const DEFAULT_FIXTURE_TIME_CONTROL: TimeControl = {
  initialTime: 600,
  increment: 0,
  type: 'blitz',
};

/**
 * Create a small backend game fixture that will be hydrated into a
 * decision-phase state (currently line_processing) when the corresponding
 * GameSession is initialised.
 *
 * This function only persists minimal metadata into the Game row –
 * geometry is applied at runtime by applyDecisionPhaseFixtureIfNeeded
 * based on the stored fixture marker. This keeps the database payloads
 * small and avoids having to persist full synthetic GameState snapshots.
 */
export async function createDecisionPhaseFixtureGame(
  options: CreateDecisionPhaseFixtureOptions
): Promise<string> {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw new Error('Database not available');
  }

  const scenario: DecisionPhaseScenario = options.scenario ?? 'line_processing';
  const isRated = options.isRated ?? true;

  const rngSeed = generateGameSeed();

  const fixtureMeta: DecisionPhaseFixtureMetadata = {
    kind: 'decision_phase_fixture',
    scenario,
    version: 1,
    ...(options.shortTimeoutMs && { shortTimeoutMs: options.shortTimeoutMs }),
    ...(options.shortWarningBeforeMs && { shortWarningBeforeMs: options.shortWarningBeforeMs }),
  };

  const initialGameState: Partial<GameState> & { fixture: DecisionPhaseFixtureMetadata } = {
    fixture: fixtureMeta,
  };

  const now = new Date();

  const game = await prisma.game.create({
    data: {
      boardType: 'square8' as PrismaBoardType,
      maxPlayers: 2,
      timeControl: DEFAULT_FIXTURE_TIME_CONTROL as unknown as Prisma.InputJsonValue,
      isRated,
      allowSpectators: true,
      status: 'active' as PrismaGameStatus,
      player1Id: options.creatorUserId,
      gameState: JSON.stringify(initialGameState),
      rngSeed,
      createdAt: now,
      updatedAt: now,
      startedAt: now,
    },
  });

  logger.info('Created decision-phase fixture game', {
    gameId: game.id,
    scenario,
    isRated,
  });

  return game.id;
}

/**
 * Apply a runtime decision-phase fixture overlay to the given GameEngine
 * when the associated Game row contains a matching fixture marker.
 *
 * This function is intentionally conservative: it only runs when a
 * decision_phase_fixture marker with a recognised scenario is present.
 * For all normal games it is a no-op.
 *
 * Returns true when a fixture overlay was applied.
 */
export function applyDecisionPhaseFixtureIfNeeded(
  engine: GameEngine,
  rawGameStateSnapshot: unknown
): boolean {
  const snapshot = (rawGameStateSnapshot || {}) as { fixture?: DecisionPhaseFixtureMetadata };
  const fixture = snapshot.fixture;

  if (!fixture || fixture.kind !== 'decision_phase_fixture') {
    return false;
  }

  if (fixture.scenario === 'line_processing') {
    seedLineProcessingDecisionPhase(engine);
    return true;
  }

  if (fixture.scenario === 'territory_processing') {
    seedTerritoryProcessingDecisionPhase(engine);
    return true;
  }

  if (fixture.scenario === 'chain_capture_choice') {
    seedChainCaptureChoiceDecisionPhase(engine);
    return true;
  }

  if (fixture.scenario === 'near_victory_elimination') {
    seedNearVictoryEliminationState(engine);
    return true;
  }

  if (fixture.scenario === 'near_victory_territory') {
    seedNearVictoryTerritoryState(engine);
    return true;
  }

  logger.warn('Unknown decision-phase fixture scenario; skipping overlay', {
    scenario: fixture.scenario,
  });
  return false;
}

/**
 * Seed a minimal overlength line configuration and enter line_processing
 * for Player 1. This mirrors the geometry used in orchestrator-focused
 * tests but is intentionally small and self-contained so it can safely
 * run at runtime in test/dev environments.
 */
function seedLineProcessingDecisionPhase(engine: GameEngine): void {
  const engineInternal = engine as unknown as GameEngineInternal;
  const state: GameState = engineInternal.gameState;
  const board = state.board;

  // Clear any existing board markers/territory to avoid interference.
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  const boardConfig = BOARD_CONFIGS[state.boardType];
  const boardSize = board.size ?? boardConfig.size;

  const activePlayerNumber = 1;
  const numPlayers = state.players.length || 2;

  // Use the effective line length threshold which accounts for 2-player
  // elevation on square8 (3 → 4). This matches the enumeration logic in
  // enumerateChooseLineRewardMoves.
  const requiredLength = getEffectiveLineLengthThreshold(state.boardType, numPlayers);
  const lineLength = requiredLength + 1;
  const rowIndex = 0;

  // Seed a simple horizontal overlength marker line for Player 1.
  for (let x = 0; x < boardSize; x += 1) {
    const key = positionToString({ x, y: rowIndex } as Position);
    board.markers.delete(key);
  }

  for (let x = 0; x < lineLength && x < boardSize; x += 1) {
    const pos: Position = { x, y: rowIndex };
    const marker: MarkerInfo = {
      player: activePlayerNumber,
      position: pos,
      type: 'regular',
    };
    board.markers.set(positionToString(pos), marker);
  }

  // Provide a simple elimination stack in the opposite corner so that
  // line-reward eliminations have a legal target once decisions are made.
  const elimPos: Position = { x: boardSize - 1, y: boardSize - 1 };
  const elimRings = [activePlayerNumber, activePlayerNumber, activePlayerNumber];
  const elimStack: RingStack = {
    position: elimPos,
    rings: elimRings,
    stackHeight: elimRings.length,
    capHeight: elimRings.length,
    controllingPlayer: activePlayerNumber,
  };
  board.stacks.set(positionToString(elimPos), elimStack);

  // Ensure the game is active and in the desired decision phase.
  state.gameStatus = 'active';
  state.currentPlayer = activePlayerNumber;
  (state as unknown as Record<string, unknown>).currentPhase = 'line_processing';
}

/**
 * Seed a minimal enclosed territory configuration and enter territory_processing
 * for Player 1. This creates a small ring formation that forms an enclosed
 * territory requiring a region order choice decision.
 *
 * The geometry places a 3x3 ring pattern in the corner of the board, where
 * the outer perimeter is controlled by Player 1 and the center is empty,
 * triggering the territory detection logic.
 */
function seedTerritoryProcessingDecisionPhase(engine: GameEngine): void {
  const engineInternal = engine as unknown as GameEngineInternal;
  const state: GameState = engineInternal.gameState;
  const board = state.board;

  // Clear any existing board markers/territory to avoid interference.
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  const activePlayerNumber = 1;

  // Create a minimal 3x3 territory enclosure in the corner.
  // Place rings in a perimeter pattern around position (1,1).
  //
  //   0 1 2 3
  // 0 R R R .
  // 1 R . R .
  // 2 R R R .
  // 3 . . . .
  //
  // The center (1,1) is empty, surrounded by Player 1 rings.

  const perimeterPositions: Position[] = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 2, y: 0 },
    { x: 0, y: 1 },
    { x: 2, y: 1 },
    { x: 0, y: 2 },
    { x: 1, y: 2 },
    { x: 2, y: 2 },
  ];

  for (const pos of perimeterPositions) {
    const key = positionToString(pos);
    // Each perimeter position gets a height-3 stack controlled by Player 1.
    const stack: RingStack = {
      position: pos,
      rings: [activePlayerNumber, activePlayerNumber, activePlayerNumber],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: activePlayerNumber,
    };
    board.stacks.set(key, stack);
  }

  // Register the enclosed territory. The center cell (1,1) is enclosed.
  const enclosedPosition: Position = { x: 1, y: 1 };
  const territoryId = `territory_p${activePlayerNumber}_0`;
  const territory: Territory = {
    spaces: [enclosedPosition],
    controllingPlayer: activePlayerNumber,
    isDisconnected: false,
  };
  board.territories.set(territoryId, territory);

  // Ensure the game is active and in the desired decision phase.
  state.gameStatus = 'active';
  state.currentPlayer = activePlayerNumber;
  const stateRecord = state as unknown as Record<string, unknown>;
  stateRecord.currentPhase = 'territory_processing';
  stateRecord.pendingTerritoryDecision = {
    territories: [territoryId],
    currentIndex: 0,
  };
}

/**
 * Seed a chain capture configuration that requires a capture order choice.
 * This creates a scenario where Player 1 has just landed on a position that
 * can capture multiple adjacent opponent stacks, triggering the chain capture
 * decision phase.
 *
 * The geometry places Player 1's marker in the center with two capturable
 * Player 2 stacks on adjacent cells.
 */
function seedChainCaptureChoiceDecisionPhase(engine: GameEngine): void {
  const engineInternal = engine as unknown as GameEngineInternal;
  const state: GameState = engineInternal.gameState;
  const board = state.board;

  // Clear any existing board markers/territory to avoid interference.
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  const activePlayerNumber = 1;
  const opponentPlayerNumber = 2;

  // Create a chain capture scenario:
  //
  //   2 3 4
  // 2 . O .
  // 3 O P O
  // 4 . O .
  //
  // P = Player 1 landing position (marker or top of stack)
  // O = Opponent stacks (height 2, controlled by Player 2)
  //
  // This creates a situation where Player 1 can capture in multiple directions.

  const landingPos: Position = { x: 3, y: 3 };
  const capturablePositions: Position[] = [
    { x: 2, y: 3 }, // left
    { x: 4, y: 3 }, // right
    { x: 3, y: 2 }, // up
    { x: 3, y: 4 }, // down
  ];

  // Place Player 1's landing position as a stack with their marker on top.
  const p1Stack: RingStack = {
    position: landingPos,
    rings: [activePlayerNumber, activePlayerNumber],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: activePlayerNumber,
  };
  board.stacks.set(positionToString(landingPos), p1Stack);

  // Place opponent stacks at capturable positions.
  for (const pos of capturablePositions) {
    const opponentStack: RingStack = {
      position: pos,
      rings: [opponentPlayerNumber, opponentPlayerNumber],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: opponentPlayerNumber,
    };
    board.stacks.set(positionToString(pos), opponentStack);
  }

  // Ensure the game is active and in the chain capture decision phase.
  state.gameStatus = 'active';
  state.currentPlayer = activePlayerNumber;
  const stateRecord = state as unknown as Record<string, unknown>;
  stateRecord.currentPhase = 'chain_capture_choice';
  stateRecord.pendingChainCapture = {
    captureFrom: landingPos,
    availableTargets: capturablePositions,
    capturedSoFar: [],
  };
}

/**
 * Seed a near-victory state where Player 1 is one capture away from winning
 * by ring elimination. This creates a scenario where Player 2 has 18 rings
 * eliminated out of 19, and Player 1 has a stack positioned to capture
 * Player 2's last ring.
 *
 * Victory condition: ringsPerPlayer eliminated rings triggers victory. For 2-player
 * on square8, each player has 18 rings (ringsPerPlayer = 18). Eliminating 18+ rings
 * from any combination of opponents triggers victory. Setting opponent to 17 eliminated
 * means one more capture wins (reaching 18).
 *
 * The board state places:
 * - Player 1 stack at (3,3) with 3 rings
 * - Player 2 single-ring stack at (4,3) - adjacent and capturable
 * - Game in 'movement' phase, Player 1's turn
 */
function seedNearVictoryEliminationState(engine: GameEngine): void {
  const engineInternal = engine as unknown as GameEngineInternal;
  const state: GameState = engineInternal.gameState;
  const board = state.board;

  // Clear any existing board markers/territory to avoid interference.
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  const activePlayerNumber = 1;
  const opponentPlayerNumber = 2;

  // Update player stats to near-victory state.
  // Player 2 has 18 rings eliminated (on square8 the victory threshold is 19).
  // Player 1 has 0 eliminated and some rings available.
  if (state.players.length >= 2) {
    // Player 1: healthy state
    state.players[0].eliminatedRings = 0;
    state.players[0].ringsInHand = 10;

    // Player 2: near-eliminated (all 18 rings from their supply removed)
    state.players[1].eliminatedRings = 18;
    state.players[1].ringsInHand = 0;
  }

  // Place Player 1's stack at (3,3) - height 3 for easy capture capability.
  const p1StackPos: Position = { x: 3, y: 3 };
  const p1Stack: RingStack = {
    position: p1StackPos,
    rings: [activePlayerNumber, activePlayerNumber, activePlayerNumber],
    stackHeight: 3,
    capHeight: 3,
    controllingPlayer: activePlayerNumber,
  };
  board.stacks.set(positionToString(p1StackPos), p1Stack);

  // Place Player 2's single-ring stack at (4,3) - adjacent to Player 1.
  // A height-3 stack can capture a height-1 stack via overtaking.
  const p2StackPos: Position = { x: 4, y: 3 };
  const p2Stack: RingStack = {
    position: p2StackPos,
    rings: [opponentPlayerNumber],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: opponentPlayerNumber,
  };
  board.stacks.set(positionToString(p2StackPos), p2Stack);

  // Set game to movement phase, Player 1's turn.
  // Player 1 can move their stack from (3,3) to (4,3) to capture Player 2's
  // last ring, triggering elimination victory.
  state.gameStatus = 'active';
  state.currentPlayer = activePlayerNumber;
  (state as unknown as Record<string, unknown>).currentPhase = 'movement';
}

/**
 * Seed a near-victory state where Player 1 is one region resolution away from
 * winning by territory control. This creates a scenario where:
 *
 * - Player 1 already controls just under 50% of board territory.
 * - A single additional territory region (one cell) is pending resolution in
 *   territory_processing.
 * - Once resolved in Player 1's favour, the backend will detect a
 *   territory_control victory.
 *
 * For simplicity and stability, we:
 * - Use BOARD_CONFIGS.square8.size (8x8) as the board.
 * - Mark 32 cells as already-collapsed territory for Player 1 (one less than
 *   the 50% threshold of 33 for territoryVictoryThreshold).
 * - Register a single pending Territory region containing one additional space,
 *   owned by Player 1, and set pendingTerritoryDecision to reference it.
 */
function seedNearVictoryTerritoryState(engine: GameEngine): void {
  const engineInternal = engine as unknown as GameEngineInternal;
  const state: GameState = engineInternal.gameState;
  const board = state.board;

  // Clear any existing board geometry to avoid interference.
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  const activePlayerNumber = 1;

  // Ensure board size is correctly set (fallback to BOARD_CONFIGS if needed).
  const boardConfig = BOARD_CONFIGS[state.boardType];
  const boardSize = board.size ?? boardConfig.size;

  // Territory victory threshold is precomputed in GameState; for square8 with
  // 64 spaces this is typically 33 (> 32). We seed exactly (threshold - 1)
  // already-controlled spaces and place the final space in a pending region.
  const threshold = state.territoryVictoryThreshold;
  const alreadyControlledCount = Math.max(threshold - 1, 1);

  // Seed collapsed territory spaces for Player 1 across the first rows.
  let placed = 0;
  for (let y = 0; y < boardSize && placed < alreadyControlledCount; y += 1) {
    for (let x = 0; x < boardSize && placed < alreadyControlledCount; x += 1) {
      const key = positionToString({ x, y } as Position);
      board.collapsedSpaces.set(key, activePlayerNumber);
      placed += 1;
    }
  }

  // Choose a distinct cell for the pending region (one that is not already
  // in collapsedSpaces). For simplicity, take the center cell.
  const pendingPos: Position = { x: Math.floor(boardSize / 2), y: Math.floor(boardSize / 2) };
  const pendingKey = positionToString(pendingPos);
  board.collapsedSpaces.delete(pendingKey);

  const territoryId = `near_victory_territory_p${activePlayerNumber}`;
  const pendingRegion: Territory = {
    spaces: [pendingPos],
    controllingPlayer: activePlayerNumber,
    isDisconnected: false,
  };
  board.territories.set(territoryId, pendingRegion);

  // Update player territory stats to match the already-collapsed spaces.
  if (state.players.length >= 1) {
    state.players[0].territorySpaces = alreadyControlledCount;
  }

  // Enter the territory_processing decision phase for Player 1 with a single
  // pending region.
  state.gameStatus = 'active';
  state.currentPlayer = activePlayerNumber;
  const stateRecord = state as unknown as Record<string, unknown>;
  stateRecord.currentPhase = 'territory_processing';
  stateRecord.pendingTerritoryDecision = {
    territories: [territoryId],
    currentIndex: 0,
  };
}
