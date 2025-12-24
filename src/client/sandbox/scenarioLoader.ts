/**
 * @fileoverview Scenario Loading Utilities - UTILITY, NOT CANONICAL
 *
 * SSoT alignment: This module provides **scenario loading utilities** for sandbox.
 * It contains no rules logic.
 *
 * This utility:
 * - Loads scenarios from contract test vectors
 * - Loads curated learning scenarios
 * - Loads/saves custom scenarios from localStorage
 * - Filters scenarios by category, board type, and search query
 *
 * DO NOT add rules logic here - this is a pure utility module.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type {
  LoadableScenario,
  ScenarioCategory,
  VectorBundle,
  ContractTestVector,
  CuratedScenarioBundle,
} from './scenarioTypes';
import { CUSTOM_SCENARIOS_STORAGE_KEY, MAX_CUSTOM_SCENARIOS } from './scenarioTypes';
import type { BoardType } from '../../shared/types/game';

/**
 * List of vector files to load.
 * These are copied from tests/fixtures/contract-vectors/v2/ to public/scenarios/vectors/
 * at build time.
 */
const VECTOR_FILES = [
  'placement.vectors.json',
  'movement.vectors.json',
  'capture.vectors.json',
  'chain_capture.vectors.json',
  'line_detection.vectors.json',
  'territory_processing.vectors.json',
  'territory.vectors.json',
];

/**
 * Convert a vector ID to a human-readable name.
 * e.g., "placement.initial.center" -> "Placement: Initial Center"
 */
function formatVectorName(id: string): string {
  const parts = id.split('.');
  if (parts.length === 0) return id;

  return parts
    .map((part, i) => {
      // Capitalize first part, format rest with spaces
      const formatted = part.replace(/_/g, ' ');
      if (i === 0) {
        return formatted.charAt(0).toUpperCase() + formatted.slice(1);
      }
      return formatted;
    })
    .join(': ');
}

/**
 * Convert a contract test vector to a loadable scenario.
 */
export function vectorToScenario(vector: ContractTestVector): LoadableScenario {
  const state = vector.input.state;
  const boardType = state.board.type as BoardType;

  return {
    id: vector.id,
    name: formatVectorName(vector.id),
    description: vector.description,
    category: vector.category as ScenarioCategory,
    tags: vector.tags || [],
    boardType,
    playerCount: state.players.length,
    createdAt: vector.createdAt,
    source: 'vector',
    state,
    suggestedMove: vector.input.move,
  };
}

/**
 * Load scenario bundles from vector files.
 * Fetches from /scenarios/vectors/ directory.
 *
 * @returns Array of loadable scenarios from all vector files
 */
export async function loadVectorScenarios(): Promise<LoadableScenario[]> {
  const scenarios: LoadableScenario[] = [];

  for (const filename of VECTOR_FILES) {
    try {
      const response = await fetch(`/scenarios/vectors/${filename}`);
      if (!response.ok) {
        console.warn(`Vector file not found: ${filename}`);
        continue;
      }

      const bundle: VectorBundle = await response.json();
      const vectorScenarios = bundle.vectors.map(vectorToScenario);
      scenarios.push(...vectorScenarios);
    } catch (err) {
      console.warn(`Failed to load vector file: ${filename}`, err);
    }
  }

  return scenarios;
}

/**
 * Load curated learning scenarios.
 * Fetches from /scenarios/curated.json.
 *
 * @returns Array of curated scenarios
 */
export async function loadCuratedScenarios(): Promise<LoadableScenario[]> {
  try {
    const response = await fetch('/scenarios/curated.json');
    if (!response.ok) {
      console.warn('Curated scenarios file not found');
      return [];
    }

    const bundle: CuratedScenarioBundle = await response.json();
    return bundle.scenarios;
  } catch (err) {
    console.warn('Failed to load curated scenarios', err);
    return [];
  }
}

/**
 * Load custom scenarios from localStorage.
 *
 * @returns Array of user-saved scenarios
 */
export function loadCustomScenarios(): LoadableScenario[] {
  try {
    const stored = localStorage.getItem(CUSTOM_SCENARIOS_STORAGE_KEY);
    if (!stored) return [];

    const scenarios = JSON.parse(stored) as LoadableScenario[];
    return scenarios;
  } catch (err) {
    console.warn('Failed to load custom scenarios from localStorage', err);
    return [];
  }
}

/**
 * Save a custom scenario to localStorage.
 * If a scenario with the same ID exists, it will be replaced.
 * Limits total stored scenarios to MAX_CUSTOM_SCENARIOS.
 *
 * @param scenario - The scenario to save
 */
export function saveCustomScenario(scenario: LoadableScenario): void {
  try {
    const existing = loadCustomScenarios();

    // Remove existing scenario with same ID, add new one at front
    const updated = [scenario, ...existing.filter((s) => s.id !== scenario.id)];

    // Limit to max scenarios
    const trimmed = updated.slice(0, MAX_CUSTOM_SCENARIOS);

    localStorage.setItem(CUSTOM_SCENARIOS_STORAGE_KEY, JSON.stringify(trimmed));
  } catch (err) {
    console.error('Failed to save custom scenario to localStorage', err);
    throw new Error('Failed to save scenario. LocalStorage may be full.');
  }
}

/**
 * Delete a custom scenario from localStorage.
 *
 * @param id - ID of the scenario to delete
 */
export function deleteCustomScenario(id: string): void {
  try {
    const existing = loadCustomScenarios();
    const filtered = existing.filter((s) => s.id !== id);
    localStorage.setItem(CUSTOM_SCENARIOS_STORAGE_KEY, JSON.stringify(filtered));
  } catch (err) {
    console.error('Failed to delete custom scenario from localStorage', err);
  }
}

/**
 * Load all scenarios from all sources.
 * Returns scenarios grouped by source type.
 */
export async function loadAllScenarios(): Promise<{
  curated: LoadableScenario[];
  vectors: LoadableScenario[];
  custom: LoadableScenario[];
}> {
  const [curated, vectors] = await Promise.all([loadCuratedScenarios(), loadVectorScenarios()]);

  const custom = loadCustomScenarios();

  return { curated, vectors, custom };
}

/**
 * Filter scenarios by various criteria.
 */
export function filterScenarios(
  scenarios: LoadableScenario[],
  filters: {
    category?: ScenarioCategory | 'all';
    boardType?: BoardType | 'all';
    searchQuery?: string;
  }
): LoadableScenario[] {
  return scenarios.filter((s) => {
    // Category filter
    if (filters.category && filters.category !== 'all' && s.category !== filters.category) {
      return false;
    }

    // Board type filter
    if (filters.boardType && filters.boardType !== 'all' && s.boardType !== filters.boardType) {
      return false;
    }

    // Search query filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      const matchesName = s.name.toLowerCase().includes(query);
      const matchesDescription = s.description.toLowerCase().includes(query);
      const matchesTags = s.tags.some((t) => t.toLowerCase().includes(query));
      if (!matchesName && !matchesDescription && !matchesTags) {
        return false;
      }
    }

    return true;
  });
}
