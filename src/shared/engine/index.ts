// =============================================================================
// CANONICAL RULES ENGINE - PUBLIC API
// =============================================================================
// This is the stable public API for the RingRift rules engine.
// Adapters (Server GameEngine, Client Sandbox, Python AI Service) should only
// import from this file.
//
// See docs/CANONICAL_ENGINE_API.md for full documentation.
//
// Design principles:
// - NARROW: Only essential functions are exported
// - STABLE: Changes inside the engine don't break adapters
// - DOMAIN-DRIVEN: Organized by game domain (Placement, Movement, Capture, etc.)
// - TYPE-SAFE: All inputs/outputs have explicit TypeScript types
// - PURE: No side effects; state passed in and returned out
// =============================================================================

// =============================================================================
// CORE TYPES (from src/shared/types/game.ts)
// =============================================================================

// Board & Position
export type {
  Position,
  BoardType,
  BoardState,
  BoardConfig,
  RingStack,
  MarkerInfo,
  MarkerType,
} from '../types/game';

// Game State
export type {
  GameState,
  GameStatus,
  GamePhase,
  Player,
  PlayerType,
  AIProfile,
  TimeControl,
  RulesOptions,
} from '../types/game';

// Moves & Actions
export type {
  Move,
  MoveType,
  MovePayload,
  LineInfo,
  Territory,
  AdjacencyType,
} from '../types/game';

// Game Result
export type { GameResult } from '../types/game';

// Player Choice System
export type {
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
} from '../types/game';

// Progress Tracking & History
export type { ProgressSnapshot, BoardSummary, GameHistoryEntry, GameTrace } from '../types/game';

// Discriminated Move Types
export type {
  MovementMove,
  BuildStackMove,
  CaptureMove,
  PlacementMove,
  SkipPlacementMove,
  SwapSidesMove,
  LineProcessingMove,
  TerritoryProcessingMove,
  LegacyMove,
  TypedMove,
} from '../types/game';

// Move Type Guards
export {
  isMovementMove,
  isBuildStackMove,
  isCaptureMove,
  isPlacementMove,
  isSpatialMove,
} from '../types/game';

// Configuration
export { BOARD_CONFIGS } from '../types/game';

// Position Utilities
export { positionToString, stringToPosition, positionsEqual } from '../types/game';

// Board Position Validation
export { isValidPosition } from './validators/utils';

// =============================================================================
// ENGINE TYPES (from src/shared/engine/types.ts)
// =============================================================================

// Validation (legacy)
export type { ValidationResult } from './types';

// Validation (unified - for new validators)
export type { ValidationOutcome } from './types';
export { ValidationErrorCode, isValidOutcome, validOutcome, invalidOutcome } from './types';

// Actions (for host-level action dispatch)
export type {
  GameAction,
  ActionType,
  BaseAction,
  PlaceRingAction,
  MoveStackAction,
  OvertakingCaptureAction,
  ContinueChainAction,
  ProcessLineAction,
  ChooseLineRewardAction,
  ProcessTerritoryAction,
  EliminateStackAction,
  SkipPlacementAction,
} from './types';

// Generics
export type { Validator, Mutator } from './types';

// =============================================================================
// DOMAIN RESULT TYPES
// =============================================================================

// Turn/Phase
export type { TurnAdvanceResult, PerTurnState, TurnLogicDelegates } from './turnLogic';

// Victory (from VictoryAggregate - victoryLogic.ts was removed Dec 2025)
export type { VictoryResult, VictoryReason } from './aggregates/VictoryAggregate';

// Placement
export type { PlacementContext, PlacementValidationResult } from './validators/PlacementValidator';

export type { PlacementApplicationOutcome } from './placementHelpers';

// Movement
export type { SimpleMoveTarget, MovementBoardAdapters } from './movementLogic';

// Capture
export type { CaptureBoardAdapters } from './captureLogic';

// Line Processing
export type { LineEnumerationOptions, LineDecisionApplicationOutcome } from './lineDecisionHelpers';

// Territory Processing
export type { TerritoryProcessingContext, TerritoryProcessingOutcome } from './territoryProcessing';

export type {
  TerritoryEnumerationOptions,
  TerritoryProcessApplicationOutcome,
  TerritoryEliminationScope,
  EliminateRingsFromStackOutcome,
} from './territoryDecisionHelpers';

// Board Views (for adapters implementing board interfaces)
export type {
  MovementBoardView,
  CaptureSegmentBoardView,
  MarkerPathHelpers,
  Direction,
} from './core';

// =============================================================================
// PLACEMENT DOMAIN
// =============================================================================
// Location: validators/PlacementValidator.ts, placementHelpers.ts
// Rule Reference: Section 4.1 - Ring Placement

// Validation (Board-Level)
export { validatePlacementOnBoard } from './validators/PlacementValidator';

// Validation (GameState-Level)
export { validatePlacement, validateSkipPlacement } from './validators/PlacementValidator';

// Application (stubs - to be implemented)
export { applyPlacementMove, evaluateSkipPlacementEligibility } from './placementHelpers';

// =============================================================================
// MOVEMENT DOMAIN
// =============================================================================
// Location: movementLogic.ts, validators/MovementValidator.ts
// Rule Reference: Section 8 - Movement

// Enumeration
export { enumerateSimpleMoveTargetsFromStack } from './movementLogic';

// Validation
export { validateMovement } from './validators/MovementValidator';

// Reachability Check (used for no-dead-placement and forced elimination)
export { hasAnyLegalMoveOrCaptureFromOnBoard } from './core';

// =============================================================================
// CAPTURE DOMAIN
// =============================================================================
// Location: captureLogic.ts, core.ts, validators/CaptureValidator.ts
// Rule Reference: Section 10 - Overtaking Capture

// Enumeration
export { enumerateCaptureMoves } from './captureLogic';

// Validation (single segment)
export { validateCaptureSegmentOnBoard } from './core';

// Validation (GameState-level)
export { validateCapture } from './validators/CaptureValidator';

// =============================================================================
// LINE DOMAIN
// =============================================================================
// Location: lineDetection.ts, lineDecisionHelpers.ts
// Rule Reference: Section 11 - Line Formation & Collapse

// Detection
export { findAllLines, findLinesForPlayer } from './lineDetection';

// Decision Enumeration
export { enumerateProcessLineMoves, enumerateChooseLineRewardMoves } from './lineDecisionHelpers';

// Application
export { applyProcessLineDecision, applyChooseLineRewardDecision } from './lineDecisionHelpers';

// =============================================================================
// TERRITORY DOMAIN
// =============================================================================
// Location: territoryDetection.ts, territoryProcessing.ts, territoryDecisionHelpers.ts
// Rule Reference: Section 12 - Territory Processing

// Detection
export { findDisconnectedRegions } from './territoryDetection';

// Alias for alternate naming convention
export { findDisconnectedRegions as detectDisconnectedRegions } from './territoryDetection';

// Processability Check
export {
  canProcessTerritoryRegion,
  filterProcessableTerritoryRegions,
  getProcessableTerritoryRegions,
} from './territoryProcessing';

// Board-Level Application
export { applyTerritoryRegion } from './territoryProcessing';

// Decision Enumeration
export {
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
} from './territoryDecisionHelpers';

// Application
export {
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
} from './territoryDecisionHelpers';

// =============================================================================
// VICTORY DOMAIN
// =============================================================================
// Location: aggregates/VictoryAggregate.ts
// Rule Reference: Section 13 - Victory Conditions
// Note: victoryLogic.ts was removed Dec 2025 - all victory logic is in VictoryAggregate

// Evaluation
export { evaluateVictory } from './aggregates/VictoryAggregate';

// Tie-breaker Helper
export { getLastActor } from './aggregates/VictoryAggregate';

// =============================================================================
// TURN MANAGEMENT
// =============================================================================
// Location: turnLogic.ts
// Rule Reference: Section 4 - Turn Structure

// Turn/Phase Progression
export { advanceTurnAndPhase } from './turnLogic';

// Action Availability Predicates (canonical implementations)
// These provide the single source of truth for "has any action?" queries
export {
  hasAnyPlacementForPlayer,
  hasAnyMovementForPlayer,
  hasAnyCaptureForPlayer,
  createDefaultTurnLogicDelegates,
} from './turnDelegateHelpers';

export type { DefaultTurnDelegatesConfig } from './turnDelegateHelpers';

// =============================================================================
// CORE UTILITIES
// =============================================================================
// Location: core.ts
// Shared helpers used across domains

// Geometry
export { getMovementDirectionsForBoardType, getPathPositions, calculateDistance } from './core';

// Direction Constants
export { SQUARE_MOORE_DIRECTIONS, HEX_DIRECTIONS } from './core';

// Stack Calculations
export { calculateCapHeight, countRingsOnBoardForPlayer, countRingsInPlayForPlayer } from './core';

// Marker Effects
export { applyMarkerEffectsAlongPathOnBoard } from './core';

// State Hashing & Debugging
export {
  hashGameState,
  hashGameStateSHA256,
  summarizeBoard,
  computeProgressSnapshot,
} from './core';

// =============================================================================
// LOCAL AI POLICY
// =============================================================================
// Location: localAIMoveSelection.ts
// Shared AI move selection policy for local/fallback AI

export type { LocalAIRng } from './localAIMoveSelection';
export { chooseLocalMoveFromCandidates } from './localAIMoveSelection';

// =============================================================================
// TERRITORY BORDERS
// =============================================================================
// Location: territoryBorders.ts
// Territory border marker enumeration

export type { TerritoryBorderMode, TerritoryBorderOptions } from './territoryBorders';
export { getBorderMarkerPositionsForRegion } from './territoryBorders';

// =============================================================================
// PLACEMENT AGGREGATE
// =============================================================================
// Consolidated placement domain module
// Location: aggregates/PlacementAggregate.ts

export type {
  PlacementContext as PlacementAggregateContext,
  PlacementValidationResult as PlacementAggregateValidationResult,
  PlacementApplicationOutcome as PlacementAggregateApplicationOutcome,
  SkipPlacementEligibilityResult,
} from './aggregates/PlacementAggregate';

export {
  // Validation
  validatePlacementOnBoard as validatePlacementOnBoardAggregate,
  validatePlacement as validatePlacementAggregate,
  validateSkipPlacement as validateSkipPlacementAggregate,
  // Enumeration
  enumeratePlacementPositions,
  evaluateSkipPlacementEligibility as evaluateSkipPlacementEligibilityAggregate,
  // Mutation
  applyPlacementOnBoard as applyPlacementOnBoardAggregate,
  mutatePlacement as mutatePlacementAggregate,
  applyPlacementMove as applyPlacementMoveAggregate,
} from './aggregates/PlacementAggregate';

// =============================================================================
// MOVEMENT AGGREGATE
// =============================================================================
// Consolidated movement domain module
// Location: aggregates/MovementAggregate.ts

export type {
  SimpleMoveTarget as MovementAggregateSimpleMoveTarget,
  MovementBoardAdapters as MovementAggregateBoardAdapters,
  SimpleMovementParams as MovementAggregateSimpleMovementParams,
  MovementApplicationOutcome as MovementAggregateApplicationOutcome,
  MovementValidationResult as MovementAggregateValidationResult,
  MovementMutationResult as MovementAggregateMutationResult,
} from './aggregates/MovementAggregate';

export {
  // Validation
  validateMovement as validateMovementAggregate,
  // Enumeration
  enumerateSimpleMoveTargetsFromStack as enumerateSimpleMoveTargetsFromStackAggregate,
  enumerateMovementTargets,
  enumerateSimpleMovesForPlayer,
  enumerateAllMovementMoves,
  // Mutation
  mutateMovement as mutateMovementAggregate,
  applySimpleMovement,
  applyMovement,
} from './aggregates/MovementAggregate';

// =============================================================================
// CAPTURE AGGREGATE
// =============================================================================
// Consolidated capture domain module
// Location: aggregates/CaptureAggregate.ts

export type {
  CaptureBoardAdapters as CaptureAggregateBoardAdapters,
  ChainCaptureStateSnapshot,
  ChainCaptureEnumerationOptions,
  ChainCaptureContinuationInfo,
  CaptureSegmentParams,
  CaptureApplicationOutcome,
  CaptureValidationResult as CaptureAggregateValidationResult,
  CaptureMutationResult as CaptureAggregateMutationResult,
} from './aggregates/CaptureAggregate';

export {
  // Re-exports from SharedCore
  validateCaptureSegmentOnBoard as validateCaptureSegmentOnBoardAggregate,
  // Validation
  validateCapture as validateCaptureAggregate,
  // Enumeration
  enumerateCaptureMoves as enumerateCaptureMovesAggregate,
  enumerateAllCaptureMoves,
  enumerateChainCaptureSegments,
  getChainCaptureContinuationInfo,
  enumerateChainCaptures,
  // Mutation
  mutateCapture as mutateCaptureAggregate,
  applyCaptureSegment,
  applyCapture,
} from './aggregates/CaptureAggregate';

// =============================================================================
// LINE AGGREGATE
// =============================================================================
// Consolidated line domain module
// Location: aggregates/LineAggregate.ts

export type {
  DetectedLine,
  LineCollapseDecision,
  LineValidationResult as LineAggregateValidationResult,
  LineMutationResult as LineAggregateMutationResult,
  LineEnumerationOptions as LineAggregateEnumerationOptions,
  LineDecisionApplicationOutcome as LineAggregateDecisionApplicationOutcome,
} from './aggregates/LineAggregate';

export {
  // Detection
  findAllLines as findAllLinesAggregate,
  findLinesForPlayer as findLinesForPlayerAggregate,
  findLinesContainingPosition,
  // Geometry helpers (for testing)
  getLineDirections,
  findLineInDirection,
  // Validation
  validateProcessLine as validateProcessLineAggregate,
  validateChooseLineReward as validateChooseLineRewardAggregate,
  validateLineDecision,
  // Enumeration
  enumerateProcessLineMoves as enumerateProcessLineMovesAggregate,
  enumerateChooseLineRewardMoves as enumerateChooseLineRewardMovesAggregate,
  enumerateLineCollapseOptions,
  // Mutation
  mutateProcessLine as mutateProcessLineAggregate,
  mutateChooseLineReward as mutateChooseLineRewardAggregate,
  applyProcessLineDecision as applyProcessLineDecisionAggregate,
  applyChooseLineRewardDecision as applyChooseLineRewardDecisionAggregate,
  applyLineCollapse,
} from './aggregates/LineAggregate';

// =============================================================================
// TERRITORY AGGREGATE
// =============================================================================
// Consolidated territory domain module
// Location: aggregates/TerritoryAggregate.ts

export type {
  DisconnectedRegion,
  TerritoryProcessingDecision,
  TerritoryValidationResult as TerritoryAggregateValidationResult,
  TerritoryMutationResult as TerritoryAggregateMutationResult,
  TerritoryProcessingContext as TerritoryAggregateProcessingContext,
  TerritoryProcessingOutcome as TerritoryAggregateProcessingOutcome,
  TerritoryEnumerationOptions as TerritoryAggregateEnumerationOptions,
  TerritoryProcessApplicationOutcome as TerritoryAggregateProcessApplicationOutcome,
  TerritoryEliminationScope as TerritoryAggregateEliminationScope,
  EliminateRingsFromStackOutcome as TerritoryAggregateEliminateRingsOutcome,
  TerritoryBorderMode as TerritoryAggregateBorderMode,
  TerritoryBorderOptions as TerritoryAggregateBorderOptions,
} from './aggregates/TerritoryAggregate';

export {
  // Detection
  findDisconnectedRegions as findDisconnectedRegionsAggregate,
  computeBorderMarkers,
  getBorderMarkerPositionsForRegion as getBorderMarkerPositionsForRegionAggregate,
  // Validation
  validateTerritoryDecision,
  validateProcessTerritory as validateProcessTerritoryAggregate,
  validateEliminateStack as validateEliminateStackAggregate,
  // Enumeration
  enumerateTerritoryDecisions,
  enumerateProcessTerritoryRegionMoves as enumerateProcessTerritoryRegionMovesAggregate,
  enumerateTerritoryEliminationMoves as enumerateTerritoryEliminationMovesAggregate,
  // Processability
  canProcessTerritoryRegion as canProcessTerritoryRegionAggregate,
  filterProcessableTerritoryRegions as filterProcessableTerritoryRegionsAggregate,
  getProcessableTerritoryRegions as getProcessableTerritoryRegionsAggregate,
  // Mutation
  applyTerritoryDecision,
  applyTerritoryRegion as applyTerritoryRegionAggregate,
  applyProcessTerritoryRegionDecision as applyProcessTerritoryRegionDecisionAggregate,
  applyEliminateRingsFromStackDecision as applyEliminateRingsFromStackDecisionAggregate,
  mutateProcessTerritory as mutateProcessTerritoryAggregate,
  mutateEliminateStack as mutateEliminateStackAggregate,
} from './aggregates/TerritoryAggregate';

// =============================================================================
// VICTORY AGGREGATE
// =============================================================================
// Consolidated victory domain module
// Location: aggregates/VictoryAggregate.ts

export type {
  VictoryReason as VictoryAggregateReason,
  VictoryResult as VictoryAggregateResult,
  VictoryEvaluationContext,
  DetailedVictoryResult,
} from './aggregates/VictoryAggregate';

export {
  // Evaluation
  evaluateVictory as evaluateVictoryAggregate,
  evaluateVictoryDetailed,
  checkLastPlayerStanding,
  checkScoreThreshold,
  isVictoryThresholdReached,
  // Queries
  getPlayerScore,
  getRemainingPlayers,
  isPlayerEliminated,
  getLastActor as getLastActorAggregate,
  getEliminatedRingCount,
  getTerritoryCount,
  getMarkerCount,
} from './aggregates/VictoryAggregate';

// =============================================================================
// RECOVERY AGGREGATE
// =============================================================================
// Recovery action for temporarily eliminated players
// Location: aggregates/RecoveryAggregate.ts
// Rule Reference: RR-CANON-R110–R115

export type {
  RecoveryOption,
  RecoverySlideMove,
  RecoverySlideTarget,
  RecoveryValidationResult,
  RecoveryApplicationOutcome,
} from './aggregates/RecoveryAggregate';

export {
  // Enumeration
  enumerateRecoverySlideTargets,
  hasAnyRecoveryMove,
  // Validation
  validateRecoverySlide,
  // Application
  applyRecoverySlide,
  // Cost calculation
  calculateRecoveryCost,
} from './aggregates/RecoveryAggregate';

// =============================================================================
// ELIMINATION AGGREGATE
// =============================================================================
// Location: aggregates/EliminationAggregate.ts
// Rule Reference: RR-CANON R022, R100, R113, R122, R145
// Single source of truth for all elimination semantics

export type {
  EliminationContext,
  EliminationReason,
  EliminationParams,
  EliminationResult,
  EliminationAuditEvent,
  StackEligibility,
} from './aggregates/EliminationAggregate';

export {
  // Core functions
  eliminateFromStack,
  isStackEligibleForElimination,
  // Enumeration
  enumerateEligibleStacks,
  hasEligibleEliminationTarget,
  // Utilities
  calculateCapHeight as calculateCapHeightElimination,
  getRingsToEliminate,
} from './aggregates/EliminationAggregate';

// =============================================================================
// GLOBAL ACTIONS &amp; ANM HELPERS
// =============================================================================
// Location: globalActions.ts
// Rule Reference: RR-CANON R200–R207, INV-ACTIVE-NO-MOVES, INV-TERMINATION

export type { GlobalLegalActionsSummary } from './globalActions';

export {
  hasTurnMaterial,
  hasGlobalPlacementAction,
  hasPhaseLocalInteractiveMove,
  hasForcedEliminationAction,
  computeGlobalLegalActionsSummary,
  isANMState,
  computeSMetric,
  computeTMetric,
  playerHasAnyRings,
} from './globalActions';

export type { ForcedEliminationOutcome, ForcedEliminationOption } from './globalActions';

export {
  applyForcedEliminationForPlayer,
  enumerateForcedEliminationOptions,
} from './globalActions';

// =============================================================================
// RULES CONFIGURATION HELPERS
// =============================================================================
// Location: rulesConfig.ts
// Rule Reference: RR-CANON-R001 (board parameters) + 2p line-length variant

export { getEffectiveLineLengthThreshold } from './rulesConfig';

// =============================================================================
// PHASE VALIDATION
// =============================================================================
// Location: phaseValidation.ts
// Declarative phase-move validation matrix
// Rule Reference: Phase/move type compatibility from RR-CANON spec

export type { MoveType as PhaseValidationMoveType } from './phaseValidation';

export {
  VALID_MOVES_BY_PHASE,
  ALWAYS_VALID_MOVES,
  isMoveValidInPhase,
  getValidMoveTypesForPhase,
  getPhasesForMoveType,
  getEliminationContextForPhase,
  canEliminateInPhase,
} from './phaseValidation';

// =============================================================================
// ORCHESTRATION TYPES
// =============================================================================
// Location: orchestration/types.ts
// Shared types for turn orchestration and decision handling

export type {
  DecisionType,
  DecisionContext,
  PendingDecision,
  ProcessTurnResult,
} from './orchestration/types';

// =============================================================================
// HISTORY HELPERS
// =============================================================================
// Location: historyHelpers.ts
// Shared history entry creation helpers for consistent GameHistoryEntry records

export type { CreateHistoryEntryOptions } from './historyHelpers';

export {
  createHistoryEntry,
  createProgressFromBoardSummary,
  appendHistoryEntryToState,
} from './historyHelpers';

// =============================================================================
// REPLAY HELPERS
// =============================================================================
// Location: replayHelpers.ts
// Shared helpers for reconstructing GameState from canonical GameRecord data

export { reconstructStateAtMove } from './replayHelpers';

// =============================================================================
// SWAP SIDES (PIE RULE) HELPERS
// =============================================================================
// Location: swapSidesHelpers.ts
// Shared swap sides eligibility and validation helpers
// Rule Reference: RR-CANON R180-R184 (Pie Rule / Swap Sides)

export {
  shouldOfferSwapSides,
  validateSwapSidesMove,
  applySwapSidesIdentitySwap,
} from './swapSidesHelpers';

// =============================================================================
// LAST-PLAYER-STANDING (LPS) TRACKING
// =============================================================================
// Location: lpsTracking.ts
// Shared LPS round tracking and victory evaluation helpers
// Rule Reference: RR-CANON R172 (Last Player Standing Victory)

export type {
  LpsTrackingState,
  LpsUpdateOptions,
  LpsEvaluationOptions,
  LpsEvaluationResult,
} from './lpsTracking';

export {
  createLpsTrackingState,
  resetLpsTrackingState,
  updateLpsTracking,
  finalizeCompletedLpsRound,
  evaluateLpsVictory,
  isLpsActivePhase,
  buildLpsVictoryResult,
  LPS_REQUIRED_CONSECUTIVE_ROUNDS,
} from './lpsTracking';

// =============================================================================
// CHAIN CAPTURE TRACKING
// =============================================================================
// Location: chainCaptureTracking.ts
// Shared chain capture state management helpers
// Rule Reference: RR-CANON R084, R085 (Chain captures)

export type { MinimalChainCaptureState, ChainCaptureEvaluation } from './chainCaptureTracking';

export {
  // State creation
  createEmptyChainCaptureState,
  createFullChainCaptureState,
  createMinimalChainCaptureState,
  // State updates
  updateChainCapturePosition,
  updateFullChainCaptureState,
  clearChainCaptureState,
  // Evaluation
  evaluateChainCaptureContinuation,
  isChainCapturePhase,
  isChainCaptureActive,
  getChainCapturePosition,
  getChainCapturePlayer,
  // High-level processing
  processChainCaptureResult,
  validateChainCaptureContinuation,
  getChainCaptureMoves,
} from './chainCaptureTracking';

// =============================================================================
// PLAYER STATE HELPERS
// =============================================================================
// Location: playerStateHelpers.ts
// Shared helpers for checking player material and action availability
// Rule Reference: RR-CANON R172 (Last Player Standing Victory)

export type { ActionAvailabilityDelegates } from './playerStateHelpers';

export {
  playerHasMaterial,
  playerControlsAnyStack,
  playerHasActiveMaterial,
  hasAnyRealAction,
  // Recovery helpers (RR-CANON-R110–R115)
  playerHasMarkers,
  countBuriedRings,
  isEligibleForRecovery,
} from './playerStateHelpers';

// =============================================================================
// BOARD MUTATION HELPERS
// =============================================================================
// Location: boardMutationHelpers.ts
// Utility functions for in-place board state mutation
// Preserves object references while replacing contents

export {
  replaceMapContents,
  replaceArrayContents,
  replaceObjectContents,
  copyBoardStateInPlace,
} from './boardMutationHelpers';

// =============================================================================
// ENGINE ERRORS
// =============================================================================
// Location: errors.ts
// Structured error types for the rules engine layer
// See also: src/shared/errors/GameDomainErrors.ts for session-level errors

export type { EngineErrorJSON } from './errors';

export {
  // Error codes
  EngineErrorCode,
  ERROR_CATEGORY_DESCRIPTIONS,
  // Error classes
  EngineError,
  RulesViolation,
  InvalidState,
  BoardConstraintViolation,
  MoveRequirementError,
  // Type guards
  isEngineError,
  isRulesViolation,
  isInvalidState,
  isBoardConstraintViolation,
  isMoveRequirementError,
  // Utilities
  wrapEngineError,
  entityNotFound,
  moveMissingField,
} from './errors';
