/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Turn Orchestration Module
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module provides the canonical processTurn orchestrator and related
 * utilities. It serves as the single entry point for turn processing,
 * delegating to domain aggregates for actual logic.
 *
 * Public API:
 * - processTurn: Synchronous turn processing
 * - processTurnAsync: Async turn processing with decision resolution
 * - validateMove: Validate a move against current state
 * - getValidMoves: Get all valid moves for current player
 * - hasValidMoves: Check if current player has any valid moves
 */

// Main orchestrator functions
export {
  processTurn,
  processTurnAsync,
  validateMove,
  getValidMoves,
  hasValidMoves,
} from './turnOrchestrator';

// Phase state machine - minimal exports for internal orchestrator use only
// NOTE: phaseStateMachine.ts is deprecated in favor of FSM-based orchestration.
// Only PhaseStateMachine and createTurnProcessingState are still used internally.
export {
  PhaseStateMachine,
  createTurnProcessingState,
  type PhaseContext,
} from './phaseStateMachine';

// Types
export type {
  ProcessTurnResult,
  PendingDecision,
  DecisionType,
  DecisionContext,
  TurnProcessingDelegates,
  AutoSelectStrategy,
  ProcessingEvent,
  ProcessingEventType,
  ProcessingMetadata,
  TerritoryResolutionOptions,
  TerritoryResolutionResult,
  ProcessedRegion,
  EliminationDecision,
  LineDetectionResult,
  DetectedLineInfo,
  LineCollapseOption,
  VictoryState,
  VictoryReason,
  PlayerScore,
  TieBreaker,
  TurnProcessingState,
  PerTurnFlags,
  // Discriminated union decision types (new in 2025-12-11 refactor)
  LineOrderDecision,
  LineRewardDecision,
  RegionOrderDecision,
  EliminationTargetDecision,
  CaptureDirectionDecision,
  ChainCaptureDecision,
  NoLineActionDecision,
  NoTerritoryActionDecision,
  NoMovementActionDecision,
  NoPlacementActionDecision,
} from './types';

// Type guards for decision types
export {
  isLineOrderDecision,
  isLineRewardDecision,
  isRegionOrderDecision,
  isEliminationTargetDecision,
  isChainCaptureDecision,
  isNoActionDecision,
} from './types';
