import type { GameState, Position, TimeControl } from '../../../shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../../../shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../../shared/engine/rulesConfig';
import { generateGameSeed } from '../../../shared/utils/rng';
import { getDatabaseClient } from '../../database/connection';
import { logger } from '../../utils/logger';
import type { GameEngine } from '../GameEngine';

export type DecisionPhaseScenario = 'line_processing';

export interface DecisionPhaseFixtureMetadata {
  kind: 'decision_phase_fixture';
  scenario: DecisionPhaseScenario;
  version: 1;
}

export interface CreateDecisionPhaseFixtureOptions {
  creatorUserId: string;
  scenario?: DecisionPhaseScenario;
  isRated?: boolean;
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
  };

  const initialGameState: Partial<GameState> & { fixture: DecisionPhaseFixtureMetadata } = {
    fixture: fixtureMeta,
  };

  const now = new Date();

  const game = await prisma.game.create({
    data: {
      boardType: 'square8' as any,
      maxPlayers: 2,
      timeControl: DEFAULT_FIXTURE_TIME_CONTROL as any,
      isRated,
      allowSpectators: true,
      status: 'active' as any,
      player1Id: options.creatorUserId,
      gameState: initialGameState as any,
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
  const engineAny = engine as any;
  const state: GameState = engineAny.gameState as GameState;
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
    const pos = { x, y: rowIndex } as Position;
    board.markers.set(positionToString(pos), {
      player: activePlayerNumber,
      position: pos,
      type: 'regular',
    } as any);
  }

  // Provide a simple elimination stack in the opposite corner so that
  // line-reward eliminations have a legal target once decisions are made.
  const elimPos: Position = { x: boardSize - 1, y: boardSize - 1 };
  const elimRings = [activePlayerNumber, activePlayerNumber, activePlayerNumber];
  board.stacks.set(positionToString(elimPos), {
    position: elimPos,
    rings: elimRings,
    stackHeight: elimRings.length,
    capHeight: elimRings.length,
    controllingPlayer: activePlayerNumber,
  } as any);

  // Ensure the game is active and in the desired decision phase.
  state.gameStatus = 'active';
  state.currentPlayer = activePlayerNumber;
  (state as any).currentPhase = 'line_processing';
}
