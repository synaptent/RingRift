import { useEffect, useMemo, useState } from 'react';
import type { PlayerChoice } from '../../shared/types/game';
import type { DecisionPhaseTimeoutWarningPayload } from '../../shared/types/websocket';

export interface UseDecisionCountdownArgs {
  /**
   * Currently pending decision choice for the active player.
   * When this becomes null or changes identity, any server
   * timeout override should be cleared.
   */
  pendingChoice: PlayerChoice | null;
  /**
   * Baseline client-side countdown in milliseconds. This is
   * typically derived from a choice deadline and maintained
   * locally by usePendingChoice.
   */
  baseTimeRemainingMs: number | null;
  /**
   * Optional server-emitted timeout warning payload providing
   * a more authoritative remainingMs for the pending decision.
   */
  timeoutWarning?: DecisionPhaseTimeoutWarningPayload | null;
}

export interface DecisionCountdownState {
  /**
   * Effective remaining time in milliseconds to display in the HUD
   * and decision UI. When a matching timeout warning is present,
   * this will never exceed either the client baseline or the
   * server-reported remainingMs, and is clamped to >= 0.
   */
  effectiveTimeRemainingMs: number | null;
  /**
   * True while we have a live server timeout warning whose metadata
   * matches the current pending choice. This is independent of
   * whether the server has actually shortened the timer relative to
   * the client baseline.
   */
  isServerOverrideActive: boolean;
  /**
   * True when the server-provided remainingMs is strictly lower than
   * the client baseline for the same pending choice. This indicates
   * that the server has effectively "capped" the local countdown.
   */
  isServerCapped: boolean;
}

function normalizeMs(value: number | null | undefined): number | null {
  if (typeof value !== 'number' || Number.isNaN(value)) return null;
  return value >= 0 ? value : 0;
}

export function useDecisionCountdown({
  pendingChoice,
  baseTimeRemainingMs,
  timeoutWarning,
}: UseDecisionCountdownArgs): DecisionCountdownState {
  const [overrideRemainingMs, setOverrideRemainingMs] = useState<number | null>(null);

  // Track server-provided remainingMs overrides when a matching
  // timeout warning is received for the current pending choice.
  useEffect(() => {
    // Clear any override when the pending choice is cleared.
    if (!pendingChoice) {
      setOverrideRemainingMs(null);
      return;
    }

    if (!timeoutWarning) {
      return;
    }

    const { playerNumber, remainingMs, choiceId } = timeoutWarning.data;

    // Only consider warnings for the same player.
    if (playerNumber !== pendingChoice.playerNumber) {
      return;
    }

    // When the warning is tied to a specific choiceId, require it
    // to match the current pending choice to avoid reusing stale
    // warnings for new decisions.
    if (choiceId && choiceId !== pendingChoice.id) {
      return;
    }

    const normalized = normalizeMs(remainingMs);
    if (normalized === null) {
      return;
    }

    setOverrideRemainingMs(normalized);
  }, [pendingChoice, timeoutWarning]);

  const { effectiveTimeRemainingMs, isServerOverrideActive, isServerCapped } = useMemo(() => {
    const base = normalizeMs(baseTimeRemainingMs);
    const override = normalizeMs(overrideRemainingMs);

    let effective: number | null = null;
    if (base != null && override != null) {
      // Never show *more* time than either the client baseline
      // or the most recent server override.
      effective = Math.min(base, override);
    } else if (override != null) {
      effective = override;
    } else {
      effective = base;
    }

    const isOverrideActive = override != null;
    const isCapped = isOverrideActive && base != null && override! < base;

    return {
      effectiveTimeRemainingMs: effective,
      isServerOverrideActive: isOverrideActive,
      isServerCapped: isCapped,
    };
  }, [baseTimeRemainingMs, overrideRemainingMs]);

  return { effectiveTimeRemainingMs, isServerOverrideActive, isServerCapped };
}
