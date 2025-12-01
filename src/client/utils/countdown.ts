export type CountdownSeverity = 'normal' | 'warning' | 'critical';

/**
 * Classify a countdown value into a coarse-grained severity bucket for
 * consistent HUD / dialog styling.
 *
 * Thresholds (in milliseconds):
 *   - normal:   > 10_000
 *   - warning:  3_000 < t <= 10_000
 *   - critical: 0 <= t <= 3_000 (and any negative values)
 */
export function getCountdownSeverity(timeRemainingMs: number | null | undefined): CountdownSeverity | null {
  if (typeof timeRemainingMs !== 'number' || Number.isNaN(timeRemainingMs)) {
    return null;
  }

  const ms = timeRemainingMs;

  if (ms > 10_000) {
    return 'normal';
  }

  if (ms > 3_000) {
    return 'warning';
  }

  // Treat zero or negative remaining time as critical: the UI should
  // still emphasise urgency/expiry even if the timer has just lapsed.
  return 'critical';
}
