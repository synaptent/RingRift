import { BoardType, Position } from '../../src/shared/types/game';

/**
 * Canonical rules/FAQ scenario definitions used by scenario-style Jest suites.
 *
 * This module is intentionally **data-only**: it does not depend on GameEngine,
 * BoardManager, or sandbox engines. Test files import these definitions and
 * decide how to realise them (backend engine, sandbox engine, parity harness,
 * etc.).
 *
 * When adding new scenarios, prefer IDs that reference the rule/FAQ section
 * they encode, e.g. `Rules_11_2_Q7_exact_length_line`.
 */

export interface RuleReference {
  /** Stable identifier tying this scenario to rules/FAQ docs. */
  id: string;
  /** Sections from ringrift_complete_rules.md or compact rules that this encodes. */
  rulesSections: string[];
  /** Optional FAQ questions this scenario is directly based on. */
  faqRefs?: string[];
}

/** Common base metadata for all rule-driven scenarios. */
export interface BaseRuleScenario {
  /** Rules/FAQ reference metadata. */
  ref: RuleReference;
  /** Human-readable description of the scenario intent. */
  description: string;
}

/**
 * Line-formation / graduated-reward scenarios (rules §11, FAQ Q7/Q22, etc.).
 *
 * These scenarios intentionally avoid hard-coding the required line length;
 * tests should derive it from BOARD_CONFIGS[boardType].lineLength so that
 * changes in board configuration do not require rewriting the matrix.
 */
export interface LineRewardRuleScenario extends BaseRuleScenario {
  kind: 'line-reward';
  /** Board type for which this scenario is defined. */
  boardType: BoardType;
  /**
   * Row index (y coordinate) on which the synthetic line should be created
   * for square boards. For non-square boards, tests may choose to map this
   * to an appropriate coordinate system.
   */
  rowIndex: number;
  /**
   * How many extra markers beyond the required line length this scenario
   * should create.
   *
   * - 0 → exact-length line
   * - >0 → overlength line (graduated reward cases)
   */
  overlengthBy: number;
}

/**
 * Non-capture movement scenarios (rules §8.2–8.3, compact rules §3.1–3.2,
 * FAQ Q2–Q3).
 */
export interface MovementRuleScenario extends BaseRuleScenario {
  kind: 'movement';
  boardType: BoardType;
  /** Height of the moving stack at origin. */
  stackHeight: number;
  /** Origin position for the moving stack. */
  origin: { x: number; y: number; z?: number };
  /**
   * Optional blocking features along movement rays, used to encode FAQ-style
   * examples where stacks and collapsed spaces block movement.
   */
  blockers?: {
    type: 'stack' | 'collapsed';
    position: { x: number; y: number; z?: number };
    controllingPlayer?: number;
    height?: number;
  }[];
}

/**
 * Chain-capture scenarios (Section 10.3, FAQ 15.3.1–15.3.2).
 */
export interface ChainCaptureMoveSpec {
  /**
   * Origin of the capturing stack for this segment. For hexagonal boards
   * this is a full cube-coordinate {@link Position} with x,y,z such that
   * x + y + z = 0; for square boards, z is omitted.
   */
  from: Position;
  /**
   * Position of the stack being overtaken in this segment.
   */
  captureTarget: Position;
  /**
   * Landing position for the capturing stack after jumping over the target.
   */
  to: Position;
}

export interface ChainCaptureStackSpec {
  /**
   * Initial position of the stack on the board. For hexagonal boards this
   * is a full cube-coordinate {@link Position}; for square boards, z is
   * omitted.
   */
  position: Position;
  /** Which player controls this stack at the start of the scenario. */
  player: number;
  /** Total height of the stack (all rings), used to derive capHeight. */
  height: number;
}

export interface ChainCaptureRuleScenario extends BaseRuleScenario {
  kind: 'chain-capture';
  boardType: BoardType;
  stacks: ChainCaptureStackSpec[];
  /**
   * Sequence of overtaking_capture segments to execute. For some scenarios
   * (e.g. the 180° reversal), the engine may also auto-complete additional
   * mandatory segments; tests assert aggregate effects rather than exact
   * landing coordinates.
   */
  moves: ChainCaptureMoveSpec[];
}

/**
 * Combined line-formation and territory scenarios (Sections 11–12, FAQ Q7,
 * Q20, Q22, Q23).
 *
 * These scenarios describe a single turn where a line is formed and then a
 * disconnected region is processed, exercising the documented ordering
 * (lines first, then regions) and elimination / territory accounting.
 */
export interface LineAndTerritoryRuleScenario extends BaseRuleScenario {
  kind: 'line-and-territory';
  boardType: BoardType;
  /**
   * Synthetic line parameters for this combined scenario. These are
   * deliberately similar to {@link LineRewardRuleScenario} so tests can
   * reuse helpers.
   */
  line: {
    /** Row index (or analogous coordinate) for constructing the line. */
    rowIndex: number;
    /** Extra markers beyond required line length; 0 = exact-length. */
    overlengthBy: number;
  };
  /**
   * Disconnected territory region affected after line processing. Tests are
   * free to realise this with more detailed board geometry.
   */
  territoryRegion: {
    /** Positions in the disconnected region credited to controllingPlayer. */
    spaces: Position[];
    /** Player who will ultimately gain territory credit for the region. */
    controllingPlayer: number;
    /** Player whose stack(s) inside the region are eliminated. */
    victimPlayer: number;
    /**
     * Height of the self-elimination stack for the controlling player
     * outside the region, used to check elimination accounting and the
     * S-invariant.
     */
    selfEliminationStackHeight: number;
    /** Position of that outside stack, if the scenario cares about it. */
    outsideStackPosition: Position;
  };
}

/**
 * Victory / termination scenarios (Sections 4.4, 13.4–13.5; FAQ Q11, Q24).
 *
 * These are high-level descriptions of late-game states used to assert
 * forced elimination behaviour and structural stalemate handling, without
 * prescribing full board geometry.
 */
export type VictoryKind = 'forced-elimination' | 'structural-stalemate';

export interface VictoryRuleScenario extends BaseRuleScenario {
  kind: 'victory';
  boardType: BoardType;
  /**
   * Flavour of late-game / termination rule this scenario exercises.
   */
  victoryKind: VictoryKind;
  /**
   * Minimal per-player ring counts. Tests may derive additional values such
   * as S-invariant deltas from these.
   */
  players: {
    playerNumber: number;
    ringsInHand: number;
    eliminatedRings: number;
  }[];
  /**
   * Optional stacks present on the board at scenario start. For structural
   * stalemate these are typically empty; for forced elimination there is at
   * least one stack controlled by the current player.
   */
  stacks?: {
    position: Position;
    player: number;
    height: number;
  }[];
}

/**
 * Territory disconnection / region-order scenarios (Section 12.2–12.3, FAQ Q20–Q23).
 *
 * These scenarios focus on when a disconnected region may be processed and
 * how multiple regions can be ordered when surfaced to the player.
 */
export interface TerritoryRuleScenario extends BaseRuleScenario {
  kind: 'territory';
  boardType: BoardType;
  /**
   * Player whose turn it is when disconnected regions are considered.
   */
  movingPlayer: number;
  /**
   * Disconnected regions visible to the moving player. Each region records
   * which player would gain credit for the collapsed spaces, which player’s
   * stacks are eliminated inside the region, and whether the moving player
   * has at least one stack/cap outside the region to satisfy the
   * self-elimination prerequisite from §12.2 / FAQ Q23.
   *
   * For some scenarios (e.g. Q23-positive), we also optionally record an
   * explicit outside stack position and height for self-elimination
   * tests. When present, these fields should match the geometry used by
   * backend and sandbox RulesMatrix tests when constructing the
   * self-elimination stack.
   */
  regions: {
    spaces: Position[];
    controllingPlayer: number;
    victimPlayer: number;
    /** True when the moving player has at least one stack/cap outside this region. */
    movingPlayerHasOutsideStack: boolean;
    /** Optional explicit outside stack position for self-elimination tests. */
    outsideStackPosition?: Position;
    /** Optional expected height for the outside stack used for self-elimination. */
    selfEliminationStackHeight?: number;
  }[];
}

/**
 * Scenarios for Section 11 (Line Formation & Collapse) and related FAQ
 * entries. These are representative, high-value examples rather than an
 * exhaustive list.
 */
export const lineRewardRuleScenarios: LineRewardRuleScenario[] = [
  {
    kind: 'line-reward',
    boardType: 'square8',
    rowIndex: 1,
    overlengthBy: 0,
    ref: {
      id: 'Rules_11_2_Q7_exact_length_line',
      rulesSections: ['§11.2'],
      faqRefs: ['Q7'],
    },
    description:
      'Exact-length line on square8: all markers collapse, one ring/cap is eliminated, and territory increases by exactly the line length.',
  },
  {
    kind: 'line-reward',
    boardType: 'square8',
    rowIndex: 2,
    overlengthBy: 1,
    ref: {
      id: 'Rules_11_3_Q22_overlength_line_option2_default',
      rulesSections: ['§11.2', '§11.3'],
      faqRefs: ['Q22'],
    },
    description:
      'Overlength line on square8: default backend behaviour with no PlayerChoice is Option 2 (minimum collapse, no elimination), preserving one marker.',
  },
  {
    kind: 'line-reward',
    boardType: 'square19',
    rowIndex: 3,
    overlengthBy: 2,
    ref: {
      id: 'Rules_11_3_Q22_overlength_line_option1_full_collapse_square19',
      rulesSections: ['§11.2', '§11.3'],
      faqRefs: ['Q22'],
    },
    description:
      'Overlength line on square19: explicit Option 1 case where the moving player collapses the entire line and eliminates one of their rings or a full cap from a controlled stack, trading rings for maximum territory.',
  },
];

/**
 * Scenarios for Section 8.2–8.3 (minimum distance and blocking) and FAQ Q2–Q3.
 * These mirror the focused cases in tests/unit/RuleEngine.movement.scenarios.test.ts
 * but expose the parameters through this shared matrix.
 */
export const movementRuleScenarios: MovementRuleScenario[] = [
  {
    kind: 'movement',
    boardType: 'square8',
    stackHeight: 2,
    origin: { x: 3, y: 3 },
    ref: {
      id: 'Rules_8_2_Q2_minimum_distance_square8',
      rulesSections: ['§8.2'],
      faqRefs: ['Q2'],
    },
    description:
      'Minimum distance on square8: a stack of height 2 at (3,3) must move at least Chebyshev distance 2; no legal moves land at distance 1.',
  },
  {
    kind: 'movement',
    boardType: 'square19',
    stackHeight: 3,
    origin: { x: 10, y: 10 },
    ref: {
      id: 'Rules_8_2_Q2_minimum_distance_square19',
      rulesSections: ['§8.2'],
      faqRefs: ['Q2'],
    },
    description:
      'Minimum distance on square19: a stack of height 3 at (10,10) must move at least Chebyshev distance 3; shorter landings are illegal.',
  },
  {
    kind: 'movement',
    boardType: 'hexagonal',
    stackHeight: 2,
    origin: { x: 0, y: 0, z: 0 },
    ref: {
      id: 'Rules_8_2_Q2_minimum_distance_hexagonal',
      rulesSections: ['§8.2'],
      faqRefs: ['Q2'],
    },
    description:
      'Minimum distance on hexagonal board: a stack of height 2 at the origin must move at least cube distance 2; shorter landings are illegal.',
  },
  {
    kind: 'movement',
    boardType: 'square8',
    stackHeight: 2,
    origin: { x: 3, y: 3 },
    ref: {
      id: 'Rules_8_2_Q2_markers_any_valid_space_beyond_square8',
      rulesSections: ['§8.2'],
      faqRefs: ['Q2', 'Q3'],
    },
    description:
      'Movement over markers on square8: a height-2 stack at (3,3) may land on any valid empty space beyond a run of markers along a ray, not just the first such space, provided the minimum-distance and path rules are met.',
  },
  {
    kind: 'movement',
    boardType: 'square8',
    stackHeight: 2,
    origin: { x: 3, y: 3 },
    ref: {
      id: 'Rules_8_2_Q2_marker_landing_own_vs_opponent_square8',
      rulesSections: ['§8.2', '§8.3'],
      faqRefs: ['Q2', 'Q3'],
    },
    description:
      'Landing on markers on square8: a height-2 stack at (3,3) is allowed to land on its own marker along a movement ray but not on an opponent marker, in line with the unified landing rule.',
  },
  {
    kind: 'movement',
    boardType: 'hexagonal',
    stackHeight: 2,
    origin: { x: 0, y: 0, z: 0 },
    ref: {
      id: 'Rules_8_2_Q2_marker_landing_own_vs_opponent_hexagonal',
      rulesSections: ['§8.2', '§8.3'],
      faqRefs: ['Q2', 'Q3'],
    },
    description:
      'Landing on markers on the hexagonal board: a height-2 stack at the origin may land on its own marker along a cube-axis movement ray but not on an opponent marker at the same minimum-distance radius, matching the unified landing rule for all board types.',
  },
  {
    kind: 'movement',
    boardType: 'square8',
    stackHeight: 2,
    origin: { x: 3, y: 3 },
    blockers: [
      {
        type: 'stack',
        position: { x: 5, y: 3 },
        controllingPlayer: 2,
        height: 1,
      },
      {
        type: 'collapsed',
        position: { x: 3, y: 5 },
      },
    ],
    ref: {
      id: 'Rules_8_3_Q3_blocked_by_stacks_and_collapsed_square8',
      rulesSections: ['§8.2', '§8.3'],
      faqRefs: ['Q2', 'Q3'],
    },
    description:
      'Movement blocking on square8: a stack at (3,3) cannot move through a blocking stack at (5,3) or a collapsed space at (3,5) along straight rays.',
  },
];

/**
 * Scenarios for Section 10.3 (Chain Overtaking) and FAQ 15.3.1–15.3.2.
 * These mirror the backend scenarios in tests/scenarios/ComplexChainCaptures.test.ts
 * but expose starting stacks and one or more scripted segments through the
 * shared rules matrix.
 */
export const chainCaptureRuleScenarios: ChainCaptureRuleScenario[] = [
  {
    kind: 'chain-capture',
    boardType: 'square19',
    stacks: [
      { position: { x: 4, y: 4 }, player: 1, height: 4 },
      { position: { x: 6, y: 4 }, player: 2, height: 3 },
    ],
    moves: [
      {
        from: { x: 4, y: 4 },
        captureTarget: { x: 6, y: 4 },
        to: { x: 8, y: 4 },
      },
    ],
    ref: {
      id: 'Rules_10_3_Q15_3_1_180_degree_reversal_basic',
      rulesSections: ['§10.3'],
      faqRefs: ['Q15.3.1'],
    },
    description:
      '180° reversal pattern on square19: Blue height-4 stack overtakes Red height-3 at B, then (via mandatory continuation) overtakes again from B, ending with Blue height 6 and Red reduced to height 1 at B.',
  },
  {
    kind: 'chain-capture',
    boardType: 'square8',
    stacks: [
      { position: { x: 3, y: 3 }, player: 1, height: 1 },
      { position: { x: 3, y: 4 }, player: 2, height: 1 },
      { position: { x: 4, y: 4 }, player: 2, height: 1 },
      { position: { x: 4, y: 3 }, player: 2, height: 1 },
    ],
    moves: [
      {
        from: { x: 3, y: 3 },
        captureTarget: { x: 3, y: 4 },
        to: { x: 3, y: 5 },
      },
      {
        from: { x: 3, y: 5 },
        captureTarget: { x: 4, y: 4 },
        to: { x: 5, y: 3 },
      },
      {
        from: { x: 5, y: 3 },
        captureTarget: { x: 4, y: 3 },
        to: { x: 2, y: 3 },
      },
    ],
    ref: {
      id: 'Rules_10_3_Q15_3_2_cyclic_pattern_triangle_loop',
      rulesSections: ['§10.3'],
      faqRefs: ['Q15.3.2'],
    },
    description:
      'Cyclic triangle pattern on square8: Blue starts at (3,3) and overtakes three Red neighbours in sequence, returning to the original file with a height-4 stack after a closed-loop chain.',
  },
  {
    kind: 'chain-capture',
    boardType: 'hexagonal',
    stacks: [
      // Outer vertex O1: overtaker stack for Player 1 (height 2).
      { position: { x: -4, y: 4 }, player: 1, height: 2 },
      // Inner triangle midpoints A, B, C: target stacks for Player 2 (height 2).
      { position: { x: 0, y: 0 }, player: 2, height: 2 }, // A
      { position: { x: 4, y: 0 }, player: 2, height: 2 }, // B (z = -4 implied)
      { position: { x: 0, y: 4 }, player: 2, height: 2 }, // C (z = -4 implied)
    ],
    moves: [
      {
        // Initial segment along one outer edge of the hex triangle:
        // O1 = (-4,4,0) → over A = (0,0,0) → O2 = (4,-4,0)
        from: { x: -4, y: 4, z: 0 },
        captureTarget: { x: 0, y: 0, z: 0 },
        to: { x: 4, y: -4, z: 0 },
      },
    ],
    ref: {
      id: 'Rules_10_3_Q15_3_x_hex_cyclic_triangle_pattern',
      rulesSections: ['§10.3'],
      faqRefs: ['Q15.3.x'],
    },
    description:
      'Hexagonal cyclic triangle pattern: Blue starts at outer vertex O1 with height 2 and overtakes Red stacks at the inner triangle midpoints A, B, C along cube-axis capture rays. The initial scripted segment O1 → A → O2 mirrors the FAQ 15.3.x-style hex cyclic example; further mandatory segments are discovered via the unified chain_capture move enumeration.',
  },
  {
    kind: 'chain-capture',
    boardType: 'square8',
    stacks: [
      { position: { x: 3, y: 3 }, player: 1, height: 1 },
      { position: { x: 3, y: 4 }, player: 2, height: 1 },
      { position: { x: 4, y: 3 }, player: 2, height: 1 },
      { position: { x: 6, y: 3 }, player: 2, height: 1 },
    ],
    moves: [
      {
        // Option A: (3,3) jumps (3,4) to land at (3,5). No further captures:
        // this demonstrates that players may choose a capture that ends the
        // chain immediately even when another available capture would permit
        // it to continue.
        from: { x: 3, y: 3 },
        captureTarget: { x: 3, y: 4 },
        to: { x: 3, y: 5 },
      },
    ],
    ref: {
      id: 'Rules_10_3_strategic_chain_ending_choice_square8',
      rulesSections: ['§10.3'],
      faqRefs: [],
    },
    description:
      'Strategic chain-ending choice on square8: Player 1 can deliberately choose a capture that leads to a position with no further legal captures, ending the mandatory chain even though a different capture from the same start position would allow it to continue.',
  },
  {
    kind: 'chain-capture',
    boardType: 'square8',
    stacks: [
      { position: { x: 0, y: 0 }, player: 1, height: 1 },
      { position: { x: 1, y: 1 }, player: 2, height: 1 },
      { position: { x: 3, y: 2 }, player: 2, height: 1 },
      { position: { x: 4, y: 3 }, player: 2, height: 1 },
    ],
    moves: [
      {
        // Start of a zig-zag chain:
        // (0,0) -> (1,1) -> (2,2). Subsequent mandatory segments may change
        // direction while still respecting straight-line geometry for each hop.
        from: { x: 0, y: 0 },
        captureTarget: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      },
    ],
    ref: {
      id: 'Rules_10_3_multi_directional_zigzag_chain_square8',
      rulesSections: ['§10.3'],
      faqRefs: [],
    },
    description:
      'Multi-directional zig-zag chain on square8: a single starting overtaking capture leads into further mandatory captures that can change direction between segments, illustrating that the chain_capture enumeration supports direction changes while preserving straight-line geometry per hop.',
  },
  {
    kind: 'chain-capture',
    boardType: 'square8',
    stacks: [
      { position: { x: 0, y: 0 }, player: 1, height: 1 },
      { position: { x: 1, y: 1 }, player: 2, height: 1 },
      { position: { x: 3, y: 3 }, player: 2, height: 1 },
      { position: { x: 5, y: 5 }, player: 2, height: 1 },
      { position: { x: 5, y: 6 }, player: 2, height: 1 },
    ],
    moves: [
      {
        // Start the chain with diagonal SE direction:
        // (0,0) -> (1,1) -> (2,2). Chain continues SE to (3,3) -> (4,4),
        // then SE to (5,5) -> (6,6), finally W to (5,6) -> (4,6).
        from: { x: 0, y: 0 },
        captureTarget: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      },
    ],
    ref: {
      id: 'Rules_10_3_chain_capture_4_targets_diagonal_with_turn',
      rulesSections: ['§10.3'],
      faqRefs: ['Q15.3.1', 'Q15.3.2'],
    },
    description:
      'Extended 4-target chain capture on square8: Blue starts at (0,0) and captures four Red targets in sequence (diagonal SE followed by W turn), resulting in a height-5 stack after completing all mandatory chain segments.',
  },
  {
    kind: 'chain-capture',
    boardType: 'square8',
    stacks: [
      { position: { x: 0, y: 0 }, player: 1, height: 1 },
      { position: { x: 1, y: 1 }, player: 2, height: 1 },
      { position: { x: 3, y: 3 }, player: 2, height: 1 },
      { position: { x: 5, y: 5 }, player: 2, height: 1 },
      { position: { x: 5, y: 6 }, player: 2, height: 1 },
      { position: { x: 3, y: 5 }, player: 2, height: 1 },
    ],
    moves: [
      {
        // Start the chain with diagonal SE direction:
        // (0,0) -> (1,1) -> (2,2). Chain continues through 5 targets total.
        from: { x: 0, y: 0 },
        captureTarget: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      },
    ],
    ref: {
      id: 'Rules_10_3_chain_capture_5_targets_extended_zigzag',
      rulesSections: ['§10.3'],
      faqRefs: ['Q15.3.1', 'Q15.3.2'],
    },
    description:
      'Extended 5-target chain capture on square8: Blue starts at (0,0) and captures five Red targets in a zigzag pattern (SE → SE → SE → W → NW), resulting in a height-6 stack at (2,4) after completing all mandatory chain segments.',
  },
];

/**
 * Scenarios where line formation and territory processing interact in a
 * single turn: an overlength line is processed first, followed by one or
 * more disconnected regions that may trigger self-elimination.
 */
export const lineAndTerritoryRuleScenarios: LineAndTerritoryRuleScenario[] = [
  {
    kind: 'line-and-territory',
    boardType: 'square8',
    line: {
      rowIndex: 0,
      overlengthBy: 1,
    },
    territoryRegion: {
      spaces: [{ x: 5, y: 5 }],
      controllingPlayer: 1,
      victimPlayer: 2,
      selfEliminationStackHeight: 2,
      outsideStackPosition: { x: 7, y: 7 },
    },
    ref: {
      id: 'Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square8',
      rulesSections: ['§11.2', '§11.3', '§12.2', '§12.3'],
      faqRefs: ['Q7', 'Q20', 'Q22', 'Q23'],
    },
    description:
      'Combined line + territory processing on square8: a single overlength line for Player 1 is processed first (defaulting to Option 2 with no ring elimination), then a one-cell disconnected region controlled by Player 1 is collapsed, eliminating an opponent stack inside and forcing one self-elimination from Player 1 while preserving the S-invariant.',
  },
  {
    kind: 'line-and-territory',
    boardType: 'square19',
    line: {
      rowIndex: 0,
      overlengthBy: 1,
    },
    territoryRegion: {
      spaces: [{ x: 5, y: 5 }],
      controllingPlayer: 1,
      victimPlayer: 2,
      selfEliminationStackHeight: 2,
      outsideStackPosition: { x: 7, y: 7 },
    },
    ref: {
      id: 'Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square19',
      rulesSections: ['§11.2', '§11.3', '§12.2', '§12.3'],
      faqRefs: ['Q7', 'Q20', 'Q22', 'Q23'],
    },
    description:
      'Same combined line + territory processing pattern as the square8 scenario but on square19, ensuring the ordering (lines before disconnected regions) and elimination accounting generalise to larger square boards.',
  },
  {
    kind: 'line-and-territory',
    boardType: 'hexagonal',
    line: {
      rowIndex: 0,
      overlengthBy: 1,
    },
    territoryRegion: {
      spaces: [{ x: 5, y: 5, z: -10 }],
      controllingPlayer: 1,
      victimPlayer: 2,
      selfEliminationStackHeight: 2,
      outsideStackPosition: { x: 7, y: 7, z: -14 },
    },
    ref: {
      id: 'Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_hexagonal',
      rulesSections: ['§11.2', '§11.3', '§12.2', '§12.3'],
      faqRefs: ['Q7', 'Q20', 'Q22', 'Q23'],
    },
    description:
      'Hexagonal analogue of the combined line + territory scenario: an overlength line for Player 1 is processed first, then a one-cell disconnected region under their control (containing an opponent stack) is collapsed, including the mandatory self-elimination for Player 1.',
  },
];

/**
 * Territory-only scenarios for Section 12 (disconnected regions) and related
 * FAQ entries (especially Q23 and region-order examples).
 */
export const territoryRuleScenarios: TerritoryRuleScenario[] = [
  {
    kind: 'territory',
    boardType: 'square19',
    movingPlayer: 1,
    regions: [
      {
        spaces: [
          { x: 5, y: 5 },
          { x: 6, y: 5 },
          { x: 7, y: 5 },
          { x: 5, y: 6 },
          { x: 6, y: 6 },
          { x: 7, y: 6 },
          { x: 5, y: 7 },
          { x: 6, y: 7 },
          { x: 7, y: 7 },
        ],
        controllingPlayer: 1,
        victimPlayer: 2,
        movingPlayerHasOutsideStack: false,
      },
    ],
    ref: {
      id: 'Rules_12_2_Q23_region_not_processed_without_self_elimination_square19',
      rulesSections: ['§12.2'],
      faqRefs: ['Q23'],
    },
    description:
      'Disconnected territory region on square19 where the moving player controls the region but has no stack or cap outside it. The self-elimination prerequisite is not satisfied, so the region must NOT be processed (no collapse, no eliminations for the moving player).',
  },
  {
    kind: 'territory',
    boardType: 'square19',
    movingPlayer: 1,
    regions: [
      {
        spaces: [
          { x: 5, y: 5 },
          { x: 6, y: 5 },
          { x: 7, y: 5 },
          { x: 5, y: 6 },
          { x: 6, y: 6 },
          { x: 7, y: 6 },
          { x: 5, y: 7 },
          { x: 6, y: 7 },
          { x: 7, y: 7 },
        ],
        controllingPlayer: 1,
        victimPlayer: 2,
        movingPlayerHasOutsideStack: true,
        // Explicit outside stack geometry for self-elimination tests:
        // a height-2 stack for the moving player at (0,1).
        outsideStackPosition: { x: 0, y: 1 },
        selfEliminationStackHeight: 2,
      },
    ],
    ref: {
      id: 'Rules_12_2_Q23_region_processed_with_self_elimination_square19',
      rulesSections: ['§12.2'],
      faqRefs: ['Q23'],
    },
    description:
      'Complementary Q23 scenario on square19: the moving player controls a disconnected region AND has at least one stack or cap outside it, so the region may be processed, collapsing all interior spaces to their colour, eliminating all rings inside, and paying the mandatory self-elimination cost from outside.',
  },
  {
    kind: 'territory',
    boardType: 'square8',
    movingPlayer: 1,
    regions: [
      {
        spaces: [
          { x: 2, y: 2 },
          { x: 2, y: 3 },
          { x: 3, y: 2 },
          { x: 3, y: 3 },
        ],
        controllingPlayer: 1,
        victimPlayer: 2,
        movingPlayerHasOutsideStack: true,
        outsideStackPosition: { x: 0, y: 0 },
        selfEliminationStackHeight: 3,
      },
    ],
    ref: {
      id: 'Rules_12_2_Q23_mini_region_square8_numeric_invariant',
      rulesSections: ['§12.2'],
      faqRefs: ['Q23'],
    },
    description:
      'Compact Q23 mini-region on square8: Player 1 controls a 2×2 disconnected region containing Player 2 stacks, has a height-3 outside stack at (0,0), and numeric invariants for territory, eliminatedRings, and the S-invariant are asserted directly at the rules layer.',
  },
  {
    kind: 'territory',
    boardType: 'square8',
    movingPlayer: 1,
    regions: [
      {
        spaces: [
          { x: 1, y: 1 },
          { x: 1, y: 2 },
        ],
        controllingPlayer: 0,
        victimPlayer: 0,
        movingPlayerHasOutsideStack: true,
      },
      {
        spaces: [
          { x: 5, y: 5 },
          { x: 5, y: 6 },
        ],
        controllingPlayer: 0,
        victimPlayer: 0,
        movingPlayerHasOutsideStack: true,
      },
    ],
    ref: {
      id: 'Rules_12_3_region_order_choice_two_regions_square8',
      rulesSections: ['§12.3'],
      faqRefs: ['Q20'],
    },
    description:
      'Two synthetic disconnected regions on square8 surfaced to the moving player in a RegionOrderChoice. Tests use this scenario to assert that when the player selects the SECOND option, the sandbox and backend engines both process that region first.',
  },
  // Hexagonal board territory scenario - fills coverage gap for hex T2 scenarios
  {
    kind: 'territory',
    boardType: 'hexagonal',
    movingPlayer: 1,
    regions: [
      {
        // Hex coords: cluster of 3 cells in a triangle pattern (q+r+s=0)
        spaces: [
          { x: -5, y: 3, z: 2 },
          { x: -4, y: 3, z: 1 },
          { x: -5, y: 4, z: 1 },
        ],
        controllingPlayer: 1,
        victimPlayer: 2,
        movingPlayerHasOutsideStack: true,
        outsideStackPosition: { x: 0, y: 0, z: 0 },
        selfEliminationStackHeight: 2,
      },
    ],
    ref: {
      id: 'Rules_12_2_Q23_hex_triangle_region_with_self_elimination',
      rulesSections: ['§12.2', '§12.3'],
      faqRefs: ['Q23'],
    },
    description:
      'Hexagonal board territory processing with self-elimination prerequisite: Player 1 controls a triangular disconnected region containing Player 2 stacks, has an outside stack at origin, and must perform mandatory self-elimination before processing the region.',
  },
  {
    kind: 'territory',
    boardType: 'hexagonal',
    movingPlayer: 1,
    regions: [
      {
        // First region: small cluster near edge
        spaces: [
          { x: -6, y: 4, z: 2 },
          { x: -5, y: 4, z: 1 },
        ],
        controllingPlayer: 0, // Neutral/shared for region-order test
        victimPlayer: 0,
        movingPlayerHasOutsideStack: true,
      },
      {
        // Second region: another cluster at opposite corner
        spaces: [
          { x: 4, y: -6, z: 2 },
          { x: 5, y: -6, z: 1 },
        ],
        controllingPlayer: 0,
        victimPlayer: 0,
        movingPlayerHasOutsideStack: true,
      },
    ],
    ref: {
      id: 'Rules_12_3_region_order_choice_two_regions_hexagonal',
      rulesSections: ['§12.3'],
      faqRefs: ['Q20'],
    },
    description:
      'Two disconnected regions on hexagonal board surfaced to the moving player in a RegionOrderChoice. Tests use this scenario to assert that when the player selects either region, the engines correctly process that region first, following the same semantics as square board region ordering.',
  },
];

/**
 * Scenarios for late-game victory / stalemate rules, focused on forced
 * elimination and structural stalemate conversions.
 */
export const victoryRuleScenarios: VictoryRuleScenario[] = [
  {
    kind: 'victory',
    boardType: 'square8',
    victoryKind: 'forced-elimination',
    players: [
      { playerNumber: 1, ringsInHand: 0, eliminatedRings: 0 },
      { playerNumber: 2, ringsInHand: 0, eliminatedRings: 0 },
    ],
    stacks: [
      {
        position: { x: 0, y: 0 },
        player: 1,
        height: 2,
      },
    ],
    ref: {
      id: 'Rules_4_4_13_5_Q24_forced_elimination_single_blocked_stack',
      rulesSections: ['§4.4', '§13.5'],
      faqRefs: ['Q24'],
    },
    description:
      'Forced elimination when the current player controls a single stack but has no legal placements, movements, captures, or chain continuations: they must eliminate the entire cap of one of their stacks, increasing eliminatedRings and the global S-invariant.',
  },
  {
    kind: 'victory',
    boardType: 'square8',
    victoryKind: 'structural-stalemate',
    players: [
      { playerNumber: 1, ringsInHand: 3, eliminatedRings: 0 },
      { playerNumber: 2, ringsInHand: 5, eliminatedRings: 0 },
    ],
    stacks: [],
    ref: {
      id: 'Rules_13_4_13_5_Q11_structural_stalemate_rings_in_hand_become_eliminated',
      rulesSections: ['§13.4', '§13.5'],
      faqRefs: ['Q11'],
    },
    description:
      'Structural stalemate with no stacks on the board and only rings in hand remaining for both players: resolveBlockedState converts ringsInHand to eliminatedRings for each player, the game becomes completed, and the S-invariant strictly increases.',
  },
  {
    kind: 'victory',
    boardType: 'square8',
    victoryKind: 'forced-elimination',
    players: [
      { playerNumber: 1, ringsInHand: 0, eliminatedRings: 0 },
      { playerNumber: 2, ringsInHand: 0, eliminatedRings: 0 },
    ],
    stacks: [],
    ref: {
      id: 'Rules_13_1_ring_elimination_threshold_square8',
      rulesSections: ['§13.1'],
      faqRefs: ['Q18', 'Q21'],
    },
    description:
      'Ring-elimination victory on square8: a player reaches the ring-elimination victory threshold (18 for 2-player square8) as eliminated rings (including rings eliminated via lines, territory, disconnected regions, forced elimination, and stalemate conversion), triggering an immediate win regardless of remaining stacks or territory.',
  },
  {
    kind: 'victory',
    boardType: 'square8',
    victoryKind: 'forced-elimination',
    players: [
      { playerNumber: 1, ringsInHand: 0, eliminatedRings: 0 },
      { playerNumber: 2, ringsInHand: 0, eliminatedRings: 0 },
    ],
    stacks: [],
    ref: {
      id: 'Rules_13_2_territory_control_threshold_square8',
      rulesSections: ['§13.2'],
      faqRefs: ['Q18', 'Q21'],
    },
    description:
      'Territory-control victory on square8: a player controls strictly more than half of all board spaces as collapsed territory in their color, satisfying the territoryVictoryThreshold and ending the game even if further eliminations or moves would be possible.',
  },
  {
    kind: 'victory',
    boardType: 'square8',
    victoryKind: 'forced-elimination',
    players: [
      { playerNumber: 1, ringsInHand: 0, eliminatedRings: 0 },
      { playerNumber: 2, ringsInHand: 0, eliminatedRings: 0 },
      { playerNumber: 3, ringsInHand: 0, eliminatedRings: 0 },
    ],
    stacks: [],
    ref: {
      id: 'Rules_13_3_last_player_standing_3p_unique_actor_square8',
      rulesSections: ['§13.3'],
      faqRefs: ['Q18', 'Q21'],
    },
    description:
      'Three-player last-player-standing plateau on square8: over a full round of turns only Player 1 has any real actions (placements, movements, or overtaking captures), while Players 2 and 3 have none. After completing the round, R172 triggers a last_player_standing victory for Player 1.',
  },
];

export type RuleScenario =
  | LineRewardRuleScenario
  | MovementRuleScenario
  | ChainCaptureRuleScenario
  | LineAndTerritoryRuleScenario
  | TerritoryRuleScenario
  | VictoryRuleScenario;

/**
 * Helper to look up a scenario by its stable rules/FAQ identifier.
 */
export function getScenarioById(id: string): RuleScenario | undefined {
  return (
    lineRewardRuleScenarios.find((s) => s.ref.id === id) ||
    movementRuleScenarios.find((s) => s.ref.id === id) ||
    chainCaptureRuleScenarios.find((s) => s.ref.id === id) ||
    lineAndTerritoryRuleScenarios.find((s) => s.ref.id === id) ||
    territoryRuleScenarios.find((s) => s.ref.id === id) ||
    victoryRuleScenarios.find((s) => s.ref.id === id)
  );
}
