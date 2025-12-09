/**
 * Shared AI Module
 *
 * This module provides shared AI utilities used by both server and client.
 *
 * @module ai
 */

export {
  // Types
  type FallbackReason,
  type FallbackContext,
  type FallbackResult,
  type FallbackDiagnostics,
  type CumulativeFallbackDiagnostics,
  // Functions
  selectFallbackMove,
  shouldAttemptFallback,
  createLocalAIRng,
  deriveFallbackSeed,
  createEmptyDiagnostics,
  updateDiagnostics,
} from './AIFallbackHandler';
