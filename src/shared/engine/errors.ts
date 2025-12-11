/**
 * Engine Domain Errors - Structured error types for the rules engine layer
 *
 * This module provides consistent error types for engine-level errors that occur
 * during move validation, state mutation, and game rule enforcement.
 *
 * Error Categories:
 * - **RulesViolation**: Invalid moves per game rules (canonical spec violations)
 * - **InvalidState**: Corrupted or unexpected game state
 * - **BoardConstraintViolation**: Geometry/topology issues with board operations
 * - **MoveRequirementError**: Missing required move fields
 *
 * Relationship to GameDomainErrors:
 * - GameDomainErrors (src/shared/errors/GameDomainErrors.ts) handles session-level errors
 *   (game not found, AI timeout, player authorization, etc.)
 * - EngineErrors handles rules-engine-level errors (invalid moves, state corruption, etc.)
 *
 * Usage:
 * ```typescript
 * import { RulesViolation, InvalidState, EngineErrorCode } from './errors';
 *
 * // Throw when a move violates game rules
 * throw new RulesViolation(
 *   EngineErrorCode.LINE_REWARD_REQUIRED,
 *   'Line length > minimum requires ChooseLineRewardAction',
 *   { lineLength: 5, minimumLength: 4 }
 * );
 *
 * // Throw when game state is corrupted
 * throw new InvalidState(
 *   EngineErrorCode.PLAYER_NOT_FOUND,
 *   'Player not found in game state',
 *   { expectedPlayer: 1 }
 * );
 * ```
 *
 * Rule Reference: RULES_CANONICAL_SPEC.md
 * @module EngineErrors
 */

// =============================================================================
// ERROR CODES
// =============================================================================

/**
 * Enumeration of all engine domain error codes.
 *
 * Error codes are prefixed by category:
 * - RULES_*: Game rule violations (canonical spec)
 * - STATE_*: Game state corruption/inconsistency
 * - BOARD_*: Board geometry/topology issues
 * - MOVE_*: Move field/type issues
 * - FSM_*: State machine transition errors
 */
export enum EngineErrorCode {
  // Rules Violations - moves that violate canonical game rules
  /** Line exceeds minimum length but no reward action provided (RR-CANON-R042) */
  RULES_LINE_REWARD_REQUIRED = 'RULES_LINE_REWARD_REQUIRED',
  /** Missing collapsed positions for MINIMUM_COLLAPSE reward (RR-CANON-R043) */
  RULES_MISSING_COLLAPSE_POSITIONS = 'RULES_MISSING_COLLAPSE_POSITIONS',
  /** Move type cannot be applied in current phase (RR-CANON phase rules) */
  RULES_PHASE_MOVE_MISMATCH = 'RULES_PHASE_MOVE_MISMATCH',
  /** Invalid recovery slide move (RR-CANON-R110-R115) */
  RULES_INVALID_RECOVERY_SLIDE = 'RULES_INVALID_RECOVERY_SLIDE',
  /** Recovery slide did not form a valid line */
  RULES_RECOVERY_LINE_INVALID = 'RULES_RECOVERY_LINE_INVALID',

  // State Errors - corrupted or unexpected game state
  /** Expected player not found in game state */
  STATE_PLAYER_NOT_FOUND = 'STATE_PLAYER_NOT_FOUND',
  /** Expected stack not found at position */
  STATE_STACK_NOT_FOUND = 'STATE_STACK_NOT_FOUND',
  /** Expected region not found in territories */
  STATE_REGION_NOT_FOUND = 'STATE_REGION_NOT_FOUND',
  /** Attacker or target stack missing for capture */
  STATE_CAPTURE_STACKS_MISSING = 'STATE_CAPTURE_STACKS_MISSING',

  // Board Constraint Violations - geometry/topology issues
  /** Unknown board type in game record */
  BOARD_UNKNOWN_TYPE = 'BOARD_UNKNOWN_TYPE',
  /** Invalid position for board geometry */
  BOARD_INVALID_POSITION = 'BOARD_INVALID_POSITION',

  // Move Requirement Errors - missing required move fields
  /** Move.from is required for this move type */
  MOVE_FROM_REQUIRED = 'MOVE_FROM_REQUIRED',
  /** Move.captureTarget is required for capture moves */
  MOVE_CAPTURE_TARGET_REQUIRED = 'MOVE_CAPTURE_TARGET_REQUIRED',
  /** Unknown action type in dispatch */
  MOVE_UNKNOWN_ACTION_TYPE = 'MOVE_UNKNOWN_ACTION_TYPE',
  /** Expected place_ring move, got different type */
  MOVE_WRONG_TYPE = 'MOVE_WRONG_TYPE',

  // FSM Errors - state machine transition issues
  /** Invalid FSM state transition */
  FSM_INVALID_TRANSITION = 'FSM_INVALID_TRANSITION',

  // Internal Errors - should never happen in correct code
  /** Assertion failed - indicates a bug */
  INTERNAL_ASSERTION_FAILED = 'INTERNAL_ASSERTION_FAILED',
  /** Not implemented yet */
  INTERNAL_NOT_IMPLEMENTED = 'INTERNAL_NOT_IMPLEMENTED',
}

/**
 * Maps error codes to human-readable category descriptions.
 */
export const ERROR_CATEGORY_DESCRIPTIONS: Record<string, string> = {
  RULES_: 'Game rule violation (see RULES_CANONICAL_SPEC.md)',
  STATE_: 'Corrupted or unexpected game state',
  BOARD_: 'Board geometry/topology constraint violation',
  MOVE_: 'Missing required move field',
  FSM_: 'Invalid state machine transition',
  INTERNAL_: 'Internal engine error (bug)',
};

// =============================================================================
// BASE ERROR CLASS
// =============================================================================

/**
 * Base class for all engine domain errors.
 *
 * Provides:
 * - Structured error code for programmatic handling
 * - Context for debugging
 * - Domain indicator for error routing
 * - Canonical rule reference (when applicable)
 */
export class EngineError extends Error {
  /** Error code for programmatic handling */
  readonly code: EngineErrorCode;

  /** Additional context for debugging */
  readonly context: Record<string, unknown>;

  /** Domain that generated the error (e.g., 'LineMutator', 'TurnOrchestrator') */
  readonly domain: string;

  /** Canonical rule reference (e.g., 'RR-CANON-R042') */
  readonly ruleRef?: string | undefined;

  /** Timestamp when error occurred */
  readonly timestamp: Date;

  constructor(
    code: EngineErrorCode,
    message: string,
    context: Record<string, unknown> = {},
    domain: string = 'Engine',
    ruleRef?: string
  ) {
    super(message);
    this.name = 'EngineError';
    this.code = code;
    this.context = context;
    this.domain = domain;
    this.ruleRef = ruleRef;
    this.timestamp = new Date();

    // Maintain proper prototype chain
    Object.setPrototypeOf(this, EngineError.prototype);
  }

  /** Get error category from code prefix */
  get category(): string {
    const prefix = this.code.split('_')[0] + '_';
    return ERROR_CATEGORY_DESCRIPTIONS[prefix] ?? 'Unknown error category';
  }

  /** Serialize to a JSON-safe object for logging/debugging */
  toJSON(): EngineErrorJSON {
    return {
      error: true,
      type: this.name,
      code: this.code,
      message: this.message,
      domain: this.domain,
      context: this.context,
      ruleRef: this.ruleRef,
      category: this.category,
      timestamp: this.timestamp.toISOString(),
    };
  }
}

/**
 * JSON representation of an EngineError.
 */
export interface EngineErrorJSON {
  error: true;
  type: string;
  code: string;
  message: string;
  domain: string;
  context: Record<string, unknown>;
  ruleRef?: string | undefined;
  category: string;
  timestamp: string;
}

// =============================================================================
// SPECIFIC ERROR CLASSES
// =============================================================================

/**
 * Error for game rule violations.
 *
 * Thrown when a move or action violates the canonical game rules
 * defined in RULES_CANONICAL_SPEC.md.
 *
 * Examples:
 * - Line exceeds minimum length but no reward action provided
 * - Move type doesn't match current phase
 * - Invalid recovery slide configuration
 */
export class RulesViolation extends EngineError {
  constructor(
    code: EngineErrorCode,
    message: string,
    context: Record<string, unknown> = {},
    domain: string = 'Rules',
    ruleRef?: string
  ) {
    super(code, message, context, domain, ruleRef);
    this.name = 'RulesViolation';
    Object.setPrototypeOf(this, RulesViolation.prototype);
  }
}

/**
 * Error for corrupted or unexpected game state.
 *
 * Thrown when expected game state elements are missing or corrupted.
 * This typically indicates either:
 * 1. A bug in the engine (state wasn't properly initialized)
 * 2. External corruption of game state
 * 3. Deserialization issues
 *
 * Examples:
 * - Player not found in game state
 * - Stack not found at expected position
 * - Region not found in territory map
 */
export class InvalidState extends EngineError {
  constructor(
    code: EngineErrorCode,
    message: string,
    context: Record<string, unknown> = {},
    domain: string = 'State'
  ) {
    super(code, message, context, domain);
    this.name = 'InvalidState';
    Object.setPrototypeOf(this, InvalidState.prototype);
  }
}

/**
 * Error for board geometry/topology violations.
 *
 * Thrown when board operations encounter invalid positions,
 * unknown board types, or topology constraint violations.
 *
 * Examples:
 * - Unknown board type in game record
 * - Position outside board bounds
 * - Invalid direction for board geometry
 */
export class BoardConstraintViolation extends EngineError {
  constructor(
    code: EngineErrorCode,
    message: string,
    context: Record<string, unknown> = {},
    domain: string = 'Board'
  ) {
    super(code, message, context, domain);
    this.name = 'BoardConstraintViolation';
    Object.setPrototypeOf(this, BoardConstraintViolation.prototype);
  }
}

/**
 * Error for missing required move fields.
 *
 * Thrown when a move is missing required fields for its type.
 * This is distinct from RulesViolation because it's about
 * move structure rather than game rules.
 *
 * Examples:
 * - Move.from missing for movement moves
 * - Move.captureTarget missing for capture moves
 */
export class MoveRequirementError extends EngineError {
  constructor(
    code: EngineErrorCode,
    message: string,
    context: Record<string, unknown> = {},
    domain: string = 'Move'
  ) {
    super(code, message, context, domain);
    this.name = 'MoveRequirementError';
    Object.setPrototypeOf(this, MoveRequirementError.prototype);
  }
}

// =============================================================================
// TYPE GUARDS
// =============================================================================

/**
 * Check if an error is an EngineError.
 */
export function isEngineError(error: unknown): error is EngineError {
  return error instanceof EngineError;
}

/**
 * Check if an error is a RulesViolation.
 */
export function isRulesViolation(error: unknown): error is RulesViolation {
  return error instanceof RulesViolation;
}

/**
 * Check if an error is an InvalidState error.
 */
export function isInvalidState(error: unknown): error is InvalidState {
  return error instanceof InvalidState;
}

/**
 * Check if an error is a BoardConstraintViolation.
 */
export function isBoardConstraintViolation(error: unknown): error is BoardConstraintViolation {
  return error instanceof BoardConstraintViolation;
}

/**
 * Check if an error is a MoveRequirementError.
 */
export function isMoveRequirementError(error: unknown): error is MoveRequirementError {
  return error instanceof MoveRequirementError;
}

// =============================================================================
// UTILITIES
// =============================================================================

/**
 * Wrap an unknown error in an EngineError.
 *
 * Useful for catching and normalizing errors at domain boundaries.
 */
export function wrapEngineError(
  error: unknown,
  domain: string = 'Engine',
  context: Record<string, unknown> = {}
): EngineError {
  if (isEngineError(error)) {
    return error;
  }

  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : undefined;

  return new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, message, {
    ...context,
    originalStack: stack,
  }, domain);
}

/**
 * Create a standard "not found" InvalidState error.
 *
 * Convenience factory for the common case of expected entities not being found.
 */
export function entityNotFound(
  entityType: 'player' | 'stack' | 'region',
  context: Record<string, unknown> = {},
  domain: string = 'State'
): InvalidState {
  const codes: Record<string, EngineErrorCode> = {
    player: EngineErrorCode.STATE_PLAYER_NOT_FOUND,
    stack: EngineErrorCode.STATE_STACK_NOT_FOUND,
    region: EngineErrorCode.STATE_REGION_NOT_FOUND,
  };

  return new InvalidState(
    codes[entityType],
    `${entityType.charAt(0).toUpperCase() + entityType.slice(1)} not found`,
    context,
    domain
  );
}

/**
 * Create a standard "missing move field" MoveRequirementError.
 *
 * Convenience factory for the common case of missing required move fields.
 */
export function moveMissingField(
  fieldName: 'from' | 'captureTarget',
  moveType: string,
  context: Record<string, unknown> = {}
): MoveRequirementError {
  const codes: Record<string, EngineErrorCode> = {
    from: EngineErrorCode.MOVE_FROM_REQUIRED,
    captureTarget: EngineErrorCode.MOVE_CAPTURE_TARGET_REQUIRED,
  };

  return new MoveRequirementError(
    codes[fieldName] ?? EngineErrorCode.MOVE_FROM_REQUIRED,
    `Move.${fieldName} is required for ${moveType} moves`,
    { fieldName, moveType, ...context },
    'Move'
  );
}
