/**
 * @fileoverview Sandbox AI Diagnostics - UTILITY, NOT CANONICAL
 *
 * SSoT alignment: This module provides **diagnostics utilities** for sandbox AI.
 * It contains no rules logic.
 *
 * This utility:
 * - Records AI decision metadata to window global for debugging
 * - Provides getSandboxAiDiagnostics() for diagnostic UIs
 * - Used by sandboxAI.ts for decision source tracking
 *
 * DO NOT add rules logic here - this is a pure utility module.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type { GameState } from '../../shared/types/game';

export type SandboxAiDecisionSource = 'service' | 'local' | 'unavailable' | 'mismatch';

export interface SandboxAiDiagnosticEntry {
  timestamp: number;
  gameId: string;
  boardType: GameState['boardType'];
  numPlayers: number;
  playerNumber: number;
  requestedDifficulty: number;
  source: SandboxAiDecisionSource;
  aiType?: string;
  difficulty?: number;
  heuristicProfileId?: string | null;
  useNeuralNet?: boolean | null;
  nnModelId?: string | null;
  nnCheckpoint?: string | null;
  nnueCheckpoint?: string | null;
  thinkingTimeMs?: number | null;
  error?: string;
}

type SandboxAiDiagnosticStore = Record<number, SandboxAiDiagnosticEntry>;

declare global {
  interface Window {
    __RINGRIFT_SANDBOX_AI_META__?: SandboxAiDiagnosticStore;
  }
}

export function recordSandboxAiDiagnostics(entry: SandboxAiDiagnosticEntry): void {
  if (typeof window === 'undefined') {
    return;
  }

  if (!window.__RINGRIFT_SANDBOX_AI_META__) {
    window.__RINGRIFT_SANDBOX_AI_META__ = {};
  }

  window.__RINGRIFT_SANDBOX_AI_META__[entry.playerNumber] = entry;
}

export function getSandboxAiDiagnostics(): SandboxAiDiagnosticStore {
  if (typeof window === 'undefined') {
    return {};
  }
  return window.__RINGRIFT_SANDBOX_AI_META__ ?? {};
}
