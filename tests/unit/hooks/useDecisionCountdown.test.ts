import { renderHook, act } from '@testing-library/react';
import { useDecisionCountdown } from '../../../src/client/hooks/useDecisionCountdown';
import type { PlayerChoice } from '../../../src/shared/types/game';
import type { DecisionPhaseTimeoutWarningPayload } from '../../../src/shared/types/websocket';

const makeChoice = (overrides: Partial<PlayerChoice> = {}): PlayerChoice => ({
  id: 'choice-1',
  gameId: 'game-1',
  playerNumber: 1,
  type: 'line_order',
  timeoutMs: 10_000,
  options: [] as any,
  ...overrides,
});

const makeWarning = (
  overrides: Partial<DecisionPhaseTimeoutWarningPayload['data']> = {}
): DecisionPhaseTimeoutWarningPayload => ({
  type: 'decision_timeout_warning',
  data: {
    playerNumber: 1,
    remainingMs: 2_500,
    choiceId: 'choice-1',
    ...overrides,
  },
});

describe('useDecisionCountdown', () => {
  it('returns base time when no override is present', () => {
    const choice = makeChoice();
    const { result } = renderHook(() =>
      useDecisionCountdown({
        pendingChoice: choice,
        baseTimeRemainingMs: 5_000,
        timeoutWarning: null,
      })
    );

    expect(result.current.effectiveTimeRemainingMs).toBe(5_000);
    expect(result.current.isServerOverrideActive).toBe(false);
    expect(result.current.isServerCapped).toBe(false);
  });

  it('applies server override and marks capped when override is lower than base', () => {
    const choice = makeChoice();
    const warning = makeWarning({ remainingMs: 2_000 });

    const { result } = renderHook(() =>
      useDecisionCountdown({
        pendingChoice: choice,
        baseTimeRemainingMs: 5_000,
        timeoutWarning: warning,
      })
    );

    expect(result.current.effectiveTimeRemainingMs).toBe(2_000);
    expect(result.current.isServerOverrideActive).toBe(true);
    expect(result.current.isServerCapped).toBe(true);
  });

  it('ignores override from a different player or choice id', () => {
    const choice = makeChoice();
    const warning = makeWarning({ playerNumber: 2, remainingMs: 1_000 });

    const { result, rerender } = renderHook(
      ({ w }: { w: DecisionPhaseTimeoutWarningPayload | null }) =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 4_000,
          timeoutWarning: w,
        }),
      { initialProps: { w: warning } }
    );

    // Different player -> ignore
    expect(result.current.effectiveTimeRemainingMs).toBe(4_000);
    expect(result.current.isServerOverrideActive).toBe(false);

    // Mismatched choice id -> ignore
    rerender({ w: makeWarning({ choiceId: 'other-choice' }) });
    expect(result.current.effectiveTimeRemainingMs).toBe(4_000);
    expect(result.current.isServerOverrideActive).toBe(false);
  });

  it('caps at the minimum of base and override when override is higher', () => {
    const choice = makeChoice();
    const warning = makeWarning({ remainingMs: 9_000 });

    const { result } = renderHook(() =>
      useDecisionCountdown({
        pendingChoice: choice,
        baseTimeRemainingMs: 6_000,
        timeoutWarning: warning,
      })
    );

    expect(result.current.effectiveTimeRemainingMs).toBe(6_000);
    expect(result.current.isServerOverrideActive).toBe(true);
    expect(result.current.isServerCapped).toBe(false);
  });

  it('normalizes negative baseline values to zero when no override is present', () => {
    const choice = makeChoice();

    const { result } = renderHook(() =>
      useDecisionCountdown({
        pendingChoice: choice,
        baseTimeRemainingMs: -500,
        timeoutWarning: null,
      })
    );

    expect(result.current.effectiveTimeRemainingMs).toBe(0);
    expect(result.current.isServerOverrideActive).toBe(false);
    expect(result.current.isServerCapped).toBe(false);
  });

  it('clamps negative server override to zero and marks as capped', () => {
    const choice = makeChoice();
    const warning = makeWarning({ remainingMs: -250 });

    const { result } = renderHook(() =>
      useDecisionCountdown({
        pendingChoice: choice,
        baseTimeRemainingMs: 1_500,
        timeoutWarning: warning,
      })
    );

    expect(result.current.effectiveTimeRemainingMs).toBe(0);
    expect(result.current.isServerOverrideActive).toBe(true);
    expect(result.current.isServerCapped).toBe(true);
  });

  it('clears server override when pending choice is cleared', () => {
    const choice = makeChoice();
    const warning = makeWarning({ remainingMs: 1_500 });

    const { result, rerender } = renderHook(
      ({ p, w }: { p: PlayerChoice | null; w: DecisionPhaseTimeoutWarningPayload | null }) =>
        useDecisionCountdown({ pendingChoice: p, baseTimeRemainingMs: 3_000, timeoutWarning: w }),
      { initialProps: { p: choice, w: warning } }
    );

    expect(result.current.effectiveTimeRemainingMs).toBe(1_500);
    expect(result.current.isServerOverrideActive).toBe(true);

    // Clear the pending choice; override should reset but baseline remains.
    rerender({ p: null, w: warning });
    expect(result.current.effectiveTimeRemainingMs).toBe(3_000);
    expect(result.current.isServerOverrideActive).toBe(false);
  });
});
