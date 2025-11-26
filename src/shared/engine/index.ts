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

// Configuration
export { BOARD_CONFIGS } from '../types/game';

// Position Utilities
export { positionToString, stringToPosition, positionsEqual } from '../types/game';

// =============================================================================
// ENGINE TYPES (from src/shared/engine/types.ts)
// =============================================================================

// Validation
export type { ValidationResult } from './types';

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

// Victory
export type { VictoryResult, VictoryReason } from './victoryLogic';

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
// Location: victoryLogic.ts
// Rule Reference: Section 13 - Victory Conditions

// Evaluation
export { evaluateVictory } from './victoryLogic';

// Tie-breaker Helper
export { getLastActor } from './victoryLogic';

// =============================================================================
// TURN MANAGEMENT
// =============================================================================
// Location: turnLogic.ts
// Rule Reference: Section 4 - Turn Structure

// Turn/Phase Progression
export { advanceTurnAndPhase } from './turnLogic';

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
export { hashGameState, summarizeBoard, computeProgressSnapshot } from './core';

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
