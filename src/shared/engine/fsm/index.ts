/**
 * FSM Module - Finite State Machine for RingRift turn phases
 */

export {
  TurnStateMachine,
  transition,
  type TurnState,
  type TurnEvent,
  type TransitionResult,
  type TransitionError,
  type Action,
  type GameContext,
  // Phase states
  type RingPlacementState,
  type MovementState,
  type CaptureState,
  type ChainCaptureState,
  type LineProcessingState,
  type TerritoryProcessingState,
  type ForcedEliminationState,
  type TurnEndState,
  type GameOverState,
  // Context types
  type DetectedLine,
  type DisconnectedRegion,
  type CaptureContext,
  type EliminationTarget,
  // Enums / helpers
  type VictoryReason,
  type Direction,
  type LineRewardChoice,
  // High-level phase completion helpers used by orchestrators / parity tooling
  type PhaseAfterLineProcessing,
  type PhaseAfterTerritoryProcessing,
  onLineProcessingComplete,
  onTerritoryProcessingComplete,
} from './TurnStateMachine';

// Adapter for bridging FSM with existing Move types
export {
  moveToEvent,
  eventToMove,
  deriveStateFromGame,
  deriveGameContext,
  validateEvent,
  getValidEvents,
  describeActionEffects,
  // FSM-based validation
  validateMoveWithFSM,
  validateMoveWithFSMAndCompare,
  isMoveTypeValidForPhase,
  type FSMValidationResult,
  // Debug logging
  setFSMDebugLogger,
  consoleFSMDebugLogger,
  type FSMDebugLogger,
  type FSMDebugContext,
  // Orchestration integration
  determineNextPhaseFromFSM,
  attemptFSMTransition,
  getCurrentFSMState,
  isFSMTerminalState,
  type PhaseTransitionContext,
  type FSMTransitionAttemptResult,
  // FSM-driven orchestration
  computeFSMOrchestration,
  compareFSMWithLegacy,
  type FSMOrchestrationResult,
  type FSMDecisionSurface,
} from './FSMAdapter';
