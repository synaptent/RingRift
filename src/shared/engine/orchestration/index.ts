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

/**
 * @deprecated Phase state machine module - DO NOT USE IN PRODUCTION CODE
 *
 * The phaseStateMachine.ts file is fully deprecated as of December 2025.
 * The FSM-based orchestration (TurnStateMachine + FSMAdapter) is now canonical.
 *
 * These exports are retained ONLY for test backward compatibility:
 * - tests/unit/phaseStateMachine.shared.test.ts
 * - tests/unit/phaseStateMachine.branchCoverage.test.ts
 * - tests/unit/ForcedElimination.phase.test.ts
 *
 * Migration path:
 * - PhaseStateMachine → ProcessingStateContainer (inline in turnOrchestrator.ts)
 * - createTurnProcessingState → createProcessingState (inline in turnOrchestrator.ts)
 * - determineNextPhase → computeFSMOrchestration from ../fsm/FSMAdapter
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md for full migration status
 * @see ../fsm/TurnStateMachine.ts for canonical FSM
 * @see ../fsm/FSMAdapter.ts for FSM integration utilities
 */
export {
  /** @deprecated Use ProcessingStateContainer in turnOrchestrator.ts instead */
  PhaseStateMachine,
  /** @deprecated Use createProcessingState in turnOrchestrator.ts instead */
  createTurnProcessingState,
  /** @deprecated Part of deprecated phaseStateMachine module */
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
