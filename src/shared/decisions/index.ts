/**
 * Shared Decision Management Module
 *
 * This module provides shared state machines and utilities for managing
 * player decisions across server and client implementations.
 *
 * @module decisions
 */

export {
  // Types
  type DecisionChoiceType,
  type DecisionChoiceKind,
  type DecisionResolutionType,
  type DecisionPhaseState,
  type DecisionIdleState,
  type DecisionPendingState,
  type DecisionWarningState,
  type DecisionExpiredState,
  type DecisionResolvedState,
  type DecisionCancelledState,
  type InitializeDecisionParams,
  type DecisionTimeoutConfig,
  // State creation
  createIdleState,
  initializeDecision,
  // Transitions
  issueWarning,
  expireDecision,
  resolveDecision,
  cancelDecision,
  clearDecision,
  // Queries
  isDecisionActive,
  isDecisionTerminal,
  getRemainingTime,
  getDecisionMetadata,
  // Timeout helpers
  DEFAULT_TIMEOUT_CONFIG,
  getTimeoutForChoice,
  getWarningScheduleTime,
  getExpiryScheduleTime,
} from './DecisionPhaseState';
