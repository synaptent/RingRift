/**
 * Type definitions for sandbox scenario loading/saving functionality.
 *
 * Scenarios allow users to:
 * 1. Load pre-built test vectors for specific game situations
 * 2. Access curated learning scenarios for beginners
 * 3. Save/load custom game states to localStorage
 * 4. Export/import scenarios as JSON files
 */

import type { BoardType, Move } from '../../shared/types/game';
import type { SerializedGameState } from '../../shared/engine/contracts/serialization';

/**
 * Categories for organizing scenarios.
 * Matches contract test vector categories plus learning/custom extensions.
 */
export type ScenarioCategory =
  | 'placement'
  | 'movement'
  | 'capture'
  | 'chain_capture'
  | 'line_processing'
  | 'territory_processing'
  | 'learning'
  | 'custom';

/**
 * Difficulty levels for curated learning scenarios.
 */
export type ScenarioDifficulty = 'beginner' | 'intermediate' | 'advanced';

/**
 * Source of a scenario for filtering and display.
 */
export type ScenarioSource = 'vector' | 'curated' | 'custom';

/**
 * Metadata for a loadable scenario.
 */
export interface ScenarioMetadata {
  /** Unique identifier for the scenario */
  id: string;
  /** Human-readable name */
  name: string;
  /** Description of the scenario and what it demonstrates */
  description: string;
  /** Category for filtering */
  category: ScenarioCategory;
  /** Difficulty level (for curated learning scenarios) */
  difficulty?: ScenarioDifficulty | undefined;
  /** Tags for search/filtering */
  tags: string[];
  /** Board type (square8, square19, hexagonal) */
  boardType: BoardType;
  /** Number of players in the scenario */
  playerCount: number;
  /** ISO timestamp when scenario was created */
  createdAt: string;
  /** Where this scenario came from */
  source: ScenarioSource;
}

/**
 * A complete scenario that can be loaded into the sandbox.
 */
export interface LoadableScenario extends ScenarioMetadata {
  /** Serialized game state to load */
  state: SerializedGameState;
  /** Optional: suggested move for learning scenarios */
  suggestedMove?: Move;
  /**
   * Optional metadata for scenarios that originate from recorded self-play
   * games. When present, the sandbox host can use this to:
   * - seed the local engine from the serialized state, and
   * - optionally drive full-move replays via the ReplayPanel using the
   *   recorded gameId (when the AI service replay DB matches the source DB).
   */
  selfPlayMeta?: {
    /** Absolute or workspace-relative path to the source SQLite database. */
    dbPath: string;
    /** Game identifier within the source database. */
    gameId: string;
    /** Total moves reported by the recorder for this game. */
    totalMoves: number;
    /**
     * Optional canonical move sequence for this game.
     * When present, the sandbox host can:
     * - Reconstruct the full game trajectory locally via ClientSandboxEngine,
     * - Populate history snapshots for the HistoryPlaybackPanel slider.
     *
     * This is populated by the Self-Play Browser when loading a game so
     * that /sandbox can replay the full self-play run without making an
     * additional round-trip to the self-play service.
     */
    moves?: Move[];
  };
}

/**
 * Bundle format for vector files.
 * Matches the structure in tests/fixtures/contract-vectors/v2/*.vectors.json
 */
export interface VectorBundle {
  version: string;
  generated: string;
  count: number;
  categories: string[];
  description: string;
  vectors: ContractTestVector[];
}

/**
 * A single contract test vector.
 * Matches the structure from the test fixtures.
 */
export interface ContractTestVector {
  id: string;
  version: string;
  category: string;
  description: string;
  tags: string[];
  source: string;
  createdAt: string;
  input: {
    state: SerializedGameState;
    move: Move;
  };
  expectedOutput: {
    status: string;
    assertions: Record<string, unknown>;
  };
}

/**
 * Bundle format for curated scenarios.
 */
export interface CuratedScenarioBundle {
  version: string;
  scenarios: LoadableScenario[];
}

/**
 * Category labels for display in UI.
 */
export const CATEGORY_LABELS: Record<ScenarioCategory, string> = {
  placement: 'Ring Placement',
  movement: 'Movement',
  capture: 'Capture',
  chain_capture: 'Chain Capture',
  line_processing: 'Line Processing',
  territory_processing: 'Territory',
  learning: 'Learning',
  custom: 'Custom',
};

/**
 * Difficulty labels for display in UI.
 */
export const DIFFICULTY_LABELS: Record<ScenarioDifficulty, string> = {
  beginner: 'Beginner',
  intermediate: 'Intermediate',
  advanced: 'Advanced',
};

/**
 * LocalStorage key for custom saved scenarios.
 */
export const CUSTOM_SCENARIOS_STORAGE_KEY = 'ringrift_custom_scenarios';

/**
 * Maximum number of custom scenarios to store in localStorage.
 */
export const MAX_CUSTOM_SCENARIOS = 20;
