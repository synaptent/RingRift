/**
 * @fileoverview Sandbox State Persistence - UTILITY, NOT CANONICAL
 *
 * SSoT alignment: This module provides **state persistence utilities** for sandbox.
 * It contains no rules logic.
 *
 * This utility:
 * - Saves current game state as custom scenarios
 * - Exports scenarios to downloadable JSON files
 * - Imports scenarios from uploaded JSON files
 * - Builds test fixtures for debugging
 *
 * DO NOT add rules logic here - this is a pure utility module.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type { GameState, BoardType, Player, TimeControl, MoveType } from '../../shared/types/game';
import type { LoadableScenario, ScenarioCategory } from './scenarioTypes';
import { serializeGameState } from '../../shared/engine/contracts/serialization';
import { computeGlobalLegalActionsSummary, evaluateVictory, isANMState } from '../../shared/engine';
import { saveCustomScenario } from './scenarioLoader';
import { createInitialGameState } from '../../shared/engine/initialState';
import {
  isLegacyMoveType,
  normalizeLegacyMoveType,
} from '../../shared/engine/legacy/legacyMoveTypes';

/**
 * Metadata to include when saving a game state.
 */
export interface SavedGameMetadata {
  /** User-provided name for the saved state */
  name: string;
  /** User-provided description */
  description?: string | undefined;
  /** Optional category override */
  category?: ScenarioCategory | undefined;
  /** Optional tags */
  tags?: string[] | undefined;
}

/**
 * Generate a unique ID for a custom scenario.
 */
function generateScenarioId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 9);
  return `custom_${timestamp}_${random}`;
}

/**
 * Infer the category based on game phase.
 */
function inferCategoryFromPhase(phase: string): ScenarioCategory {
  switch (phase) {
    case 'ring_placement':
      return 'placement';
    case 'movement':
      return 'movement';
    case 'chain_capture':
      return 'chain_capture';
    case 'line_processing':
      return 'line_processing';
    case 'territory_processing':
      return 'territory_processing';
    case 'forced_elimination':
      // Forced elimination is a special 7th phase (RR-CANON-R070) that doesn't
      // map to existing scenario categories; treat as custom for now.
      return 'custom';
    default:
      break;
  }

  const maybeMoveType = phase as MoveType;
  const normalized = normalizeLegacyMoveType(maybeMoveType);

  if (normalized === 'process_line' || normalized === 'choose_line_option') {
    return 'line_processing';
  }
  if (normalized === 'choose_territory_option') {
    return 'territory_processing';
  }

  if (isLegacyMoveType(maybeMoveType)) {
    if (maybeMoveType === 'line_formation') {
      return 'line_processing';
    }
    if (maybeMoveType === 'territory_claim') {
      return 'territory_processing';
    }
  }

  return 'custom';
}

/**
 * Save the current game state as a custom scenario.
 *
 * @param gameState - The current game state to save
 * @param metadata - User-provided metadata for the saved state
 * @returns The created LoadableScenario
 */
export function saveCurrentGameState(
  gameState: GameState,
  metadata: SavedGameMetadata
): LoadableScenario {
  const serializedState = serializeGameState(gameState);
  const id = generateScenarioId();

  const turnNumber = (gameState.moveHistory?.length ?? 0) + 1;
  const defaultDescription = `Saved at turn ${turnNumber}, ${gameState.currentPhase} phase`;

  const scenario: LoadableScenario = {
    id,
    name: metadata.name || `Saved Game - ${new Date().toLocaleDateString()}`,
    description: metadata.description || defaultDescription,
    category: metadata.category || inferCategoryFromPhase(gameState.currentPhase),
    tags: metadata.tags || ['saved', gameState.currentPhase],
    boardType: gameState.boardType,
    playerCount: gameState.players.length,
    createdAt: new Date().toISOString(),
    source: 'custom',
    state: serializedState,
  };

  saveCustomScenario(scenario);
  return scenario;
}

/**
 * Export a scenario to a downloadable JSON file.
 *
 * @param scenario - The scenario to export
 */
export function exportScenarioToFile(scenario: LoadableScenario): void {
  const json = JSON.stringify(scenario, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  // Create safe filename
  const safeName = scenario.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '');
  const filename = `ringrift_scenario_${safeName}.json`;

  // Create download link and trigger click
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  // Cleanup
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export the current game state directly to a file.
 *
 * @param gameState - The game state to export
 * @param name - Name for the exported file
 */
export function exportGameStateToFile(gameState: GameState, name: string): void {
  const scenario = saveCurrentGameState(gameState, { name });
  exportScenarioToFile(scenario);
}

/**
 * Build a lightweight, test-friendly fixture object from a GameState.
 * This is intended for copying into Jest tests so that bugs discovered
 * via the sandbox UI can be reproduced as scripted fixtures.
 */
export function buildTestFixtureFromGameState(gameState: GameState): unknown {
  const serializedState = serializeGameState(gameState);

  // Global legal-actions summary per player (placements, moves, forced
  // elimination) and ANM state are extremely helpful when debugging
  // early-completion and "no legal moves but game still active" bugs.
  const perPlayerActions = gameState.players.map((p) => ({
    playerNumber: p.playerNumber,
    summary: computeGlobalLegalActionsSummary(gameState, p.playerNumber),
  }));

  const victoryProbe = evaluateVictory(gameState);
  const isActiveNoMoves = isANMState(gameState);

  return {
    kind: 'ringrift_sandbox_fixture_v1',
    boardType: gameState.boardType,
    currentPhase: gameState.currentPhase,
    currentPlayer: gameState.currentPlayer,
    rngSeed: (gameState as GameState & { rngSeed?: number | null }).rngSeed ?? null,
    moveHistory: gameState.moveHistory ?? [],
    historyLength: gameState.history?.length ?? 0,
    debug: {
      gameStatus: gameState.gameStatus,
      perPlayerActions,
      isANMState: isActiveNoMoves,
      victoryProbe,
    },
    state: serializedState,
  };
}

/**
 * Validation result for imported scenarios.
 */
export interface ScenarioValidationResult {
  valid: boolean;
  errors: string[];
  scenario?: LoadableScenario;
}

/**
 * Validate an imported scenario object.
 */
function validateScenario(data: unknown): ScenarioValidationResult {
  const errors: string[] = [];

  if (!data || typeof data !== 'object') {
    return { valid: false, errors: ['Invalid scenario format: not an object'] };
  }

  const obj = data as Record<string, unknown>;

  // Support importing sandbox test fixtures (`ringrift_sandbox_fixture_v1`)
  // directly as scenarios. These are the objects produced by
  // buildTestFixtureFromGameState and copied via the "Copy test fixture"
  // control in the sandbox UI. For these fixtures we also attach a synthetic
  // selfPlayMeta.moves array so the sandbox host can reconstruct full history
  // snapshots by replaying the canonical move sequence from an inferred
  // initial state.
  /* eslint-disable @typescript-eslint/no-explicit-any -- parsing untyped JSON fixtures */
  if (obj.kind === 'ringrift_sandbox_fixture_v1') {
    const fixture = obj as any;
    const fixtureState = fixture.state as Record<string, unknown> | undefined;
    const serializedState = fixtureState as
      | (ScenarioValidationResult['scenario'] extends { state: infer S } ? S : never)
      | undefined;

    const boardType =
      (fixture.boardType as BoardType | undefined) ||
      ((fixtureState?.board as { type?: string } | undefined)?.type as BoardType | undefined);

    const fixtureMoves: unknown[] = Array.isArray(fixture.moveHistory)
      ? (fixture.moveHistory as unknown[])
      : Array.isArray((serializedState as any)?.moveHistory)
        ? (((serializedState as any).moveHistory as unknown[]) ?? [])
        : [];

    const baseId =
      typeof fixture.id === 'string' && fixture.id.trim()
        ? (fixture.id as string)
        : 'sandbox_fixture';

    // Build minimal player stubs for initial-state reconstruction metadata.
    const serializedPlayers: Array<{ playerNumber: number }> =
      (serializedState && (serializedState as any).players) || [];
    const playerCount = serializedPlayers.length || 2;

    // Use a simple rapid time control and human players; this metadata is
    // only used to seed a structurally valid initial GameState for replay.
    const timeControl: TimeControl = {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    };

    const players: Player[] = [];
    for (let i = 0; i < playerCount; i += 1) {
      const playerNumber = i + 1;
      players.push({
        id: `player-${playerNumber}`,
        username: `Player ${playerNumber}`,
        type: 'human',
        playerNumber,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      });
    }

    if (!boardType) {
      return { valid: false, errors: ['Invalid fixture: missing boardType'] };
    }

    const initialState = createInitialGameState(
      baseId,
      boardType,
      players,
      timeControl,
      false,
      (fixture.rngSeed as number | undefined) || undefined
    );
    const initialSerializedState = serializeGameState(initialState);

    const tags =
      Array.isArray(fixture.tags) && fixture.tags.length > 0
        ? (fixture.tags as unknown[]).map(String)
        : ['sandbox', 'fixture'];

    const name =
      typeof fixture.name === 'string' && fixture.name.trim()
        ? (fixture.name as string)
        : 'Sandbox test fixture';

    const currentPhase =
      (fixture.currentPhase as string | undefined) ||
      ((serializedState as any)?.currentPhase as string | undefined);

    const description =
      typeof fixture.description === 'string' && fixture.description.trim()
        ? (fixture.description as string)
        : `Imported sandbox fixture (phase ${currentPhase ?? 'unknown'})`;

    const scenario: LoadableScenario = {
      id: `imported_${Date.now()}_${baseId}`,
      name,
      description,
      category: 'custom',
      tags,
      boardType,
      playerCount,
      createdAt: new Date().toISOString(),
      source: 'custom',
      state: initialSerializedState as LoadableScenario['state'],
      selfPlayMeta: {
        dbPath: 'sandbox_fixture',
        gameId: baseId,
        totalMoves: fixtureMoves.length,

        moves: fixtureMoves as any[],
      },
    };

    return { valid: true, errors: [], scenario };
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */

  // Required fields
  if (!obj.id || typeof obj.id !== 'string') {
    errors.push('Missing or invalid "id" field');
  }
  if (!obj.name || typeof obj.name !== 'string') {
    errors.push('Missing or invalid "name" field');
  }
  if (!obj.state || typeof obj.state !== 'object') {
    errors.push('Missing or invalid "state" field');
  }
  if (!obj.boardType || typeof obj.boardType !== 'string') {
    errors.push('Missing or invalid "boardType" field');
  }

  // Validate state structure
  if (obj.state && typeof obj.state === 'object') {
    const state = obj.state as Record<string, unknown>;
    if (!state.board || typeof state.board !== 'object') {
      errors.push('Invalid state: missing "board" field');
    }
    if (!state.players || !Array.isArray(state.players)) {
      errors.push('Invalid state: missing "players" array');
    }
    if (typeof state.currentPlayer !== 'number') {
      errors.push('Invalid state: missing "currentPlayer" field');
    }
    if (!state.currentPhase || typeof state.currentPhase !== 'string') {
      errors.push('Invalid state: missing "currentPhase" field');
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors };
  }

  // Construct validated scenario
  const scenario: LoadableScenario = {
    id: `imported_${Date.now()}_${String(obj.id)}`,
    name: String(obj.name),
    description: String(obj.description || ''),
    category: (obj.category as ScenarioCategory) || 'custom',
    difficulty: obj.difficulty as LoadableScenario['difficulty'],
    tags: Array.isArray(obj.tags) ? obj.tags.map(String) : [],
    boardType: obj.boardType as BoardType,
    playerCount: typeof obj.playerCount === 'number' ? obj.playerCount : 2,
    createdAt: new Date().toISOString(),
    source: 'custom',
    state: obj.state as LoadableScenario['state'],
  };

  return { valid: true, errors: [], scenario };
}

/**
 * Import a scenario from a JSON file.
 *
 * @param file - The uploaded File object
 * @returns Promise resolving to the imported scenario or throwing on error
 */
export async function importScenarioFromFile(file: File): Promise<LoadableScenario> {
  const text = await file.text();

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('Invalid JSON file');
  }

  const result = validateScenario(data);
  if (!result.valid || !result.scenario) {
    throw new Error(`Invalid scenario file:\n${result.errors.join('\n')}`);
  }

  return result.scenario;
}

/**
 * Import a scenario from a JSON file and save it to localStorage.
 *
 * @param file - The uploaded File object
 * @returns Promise resolving to the saved scenario
 */
export async function importAndSaveScenarioFromFile(file: File): Promise<LoadableScenario> {
  const scenario = await importScenarioFromFile(file);
  saveCustomScenario(scenario);
  return scenario;
}
