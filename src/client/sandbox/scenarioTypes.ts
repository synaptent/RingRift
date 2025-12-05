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
 * High-level rules concept families for curated scenarios.
 * These are used for metadata validation and mapping curated
 * sandbox scenarios back to canonical UX rules copy sections.
 */
export type ScenarioRulesConcept =
  | 'board_intro_square8'
  | 'board_intro_hex'
  | 'placement_basic'
  | 'movement_basic'
  | 'stack_height_mobility'
  | 'capture_basic'
  | 'chain_capture_mandatory'
  | 'chain_capture_extended'
  | 'lines_basic'
  | 'lines_overlength_option2'
  | 'territory_basic'
  | 'territory_multi_region'
  | 'territory_mini_region_q23'
  | 'territory_near_victory'
  | 'victory_ring_elimination'
  | 'victory_territory'
  | 'turn_multi_phase'
  | 'puzzle_capture'
  | 'anm_forced_elimination'
  | 'anm_last_player_standing';

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
  /**
   * Canonical rules concept this scenario primarily illustrates.
   * Used for metadata validation and mapping back to UX rules copy.
   */
  rulesConcept?: ScenarioRulesConcept;
  /**
   * Optional anchor into UX_RULES_COPY_SPEC.md (or related docs)
   * describing this scenario’s primary rules concept; e.g.
   * "movement.semantics" or "territory.q23_mini_region".
   * This is metadata-only and not parsed at runtime.
   */
  uxSpecAnchor?: string;
  /**
   * Optional flag indicating that this scenario is part of the
   * player-facing onboarding set. Used to surface a small, curated
   * subset of FAQ/rules-aligned scenarios in the sandbox UI by
   * default, without exposing the full diagnostics catalog.
   */
  onboarding?: boolean;
  /**
   * Optional short rules snippet shown inline in player-facing
   * overlays (ScenarioPicker, sandbox sidebar, onboarding helpers).
   * This is a plain-text summary derived from ringrift_compact_rules
   * and RULES_SCENARIO_MATRIX – no Markdown parsing at runtime.
   */
  rulesSnippet?: string;
  /**
   * Optional reference back to the canonical rules or matrix entry
   * that this scenario is derived from, e.g.
   * "ringrift_compact_rules#territory" or
   * "RULES_SCENARIO_MATRIX#T3".
   * This is metadata-only and never parsed at runtime.
   */
  rulesSnippetRef?: string;
  /**
   * Optional identifier for the associated RULES_SCENARIO_MATRIX
   * row or contract vector id. Kept as metadata so developers can
   * trace curated onboarding scenarios back to the underlying
   * Jest suites and vectors.
   */
  matrixScenarioId?: string;
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

/**
 * Canonical list of allowed rulesConcept values for curated scenarios.
 * Exported for use in metadata validation tests.
 */
export const SCENARIO_RULES_CONCEPTS: readonly ScenarioRulesConcept[] = [
  'board_intro_square8',
  'board_intro_hex',
  'placement_basic',
  'movement_basic',
  'stack_height_mobility',
  'capture_basic',
  'chain_capture_mandatory',
  'chain_capture_extended',
  'lines_basic',
  'lines_overlength_option2',
  'territory_basic',
  'territory_multi_region',
  'territory_mini_region_q23',
  'territory_near_victory',
  'victory_ring_elimination',
  'victory_territory',
  'turn_multi_phase',
  'puzzle_capture',
  'anm_forced_elimination',
  'anm_last_player_standing',
] as const;
