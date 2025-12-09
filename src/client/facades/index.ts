/**
 * Client Facades Module
 *
 * This module provides unified abstractions for game host implementations,
 * enabling shared UI components to work consistently across backend-connected
 * and local sandbox games.
 *
 * @module facades
 */

// Main facade interface and utilities
export {
  // Types
  type PartialMove,
  type FacadeConnectionStatus,
  type GameFacadeMode,
  type FacadeDecisionState,
  type FacadePlayerInfo,
  type FacadeChainCaptureState,
  type GameFacade,
  // Utilities
  extractChainCapturePath,
  deriveMustMoveFrom,
  canSubmitMove,
  canInteract,
} from './GameFacade';

// View model derivation hook
export {
  type GamePlayViewModelOptions,
  type GamePlayViewModels,
  useGamePlayViewModels,
  useInstructionText,
} from './useGamePlayViewModels';

// Cell interaction hook
export {
  type CellInteractionOptions,
  type CellInteractionState,
  type CellInteractionHandlers,
  useCellInteractions,
} from './useCellInteractions';
