import { getCountdownSeverity } from '../../../src/client/utils/countdown';

describe('getCountdownSeverity', () => {
  it('returns null for non-numeric or null input', () => {
    expect(getCountdownSeverity(null)).toBeNull();
    expect(getCountdownSeverity(undefined)).toBeNull();
    expect(getCountdownSeverity(NaN as unknown as number)).toBeNull();
  });

  it('returns "normal" when time remaining is greater than 10 seconds', () => {
    expect(getCountdownSeverity(11_000)).toBe('normal');
    expect(getCountdownSeverity(20_000)).toBe('normal');
  });

  it('returns "warning" when time remaining is between 3 and 10 seconds (exclusive lower bound, inclusive upper bound)', () => {
    expect(getCountdownSeverity(10_000)).toBe('warning');
    expect(getCountdownSeverity(3_001)).toBe('warning');
  });

  it('returns "critical" when time remaining is at or below 3 seconds, including zero and negative values', () => {
    expect(getCountdownSeverity(3_000)).toBe('critical');
    expect(getCountdownSeverity(0)).toBe('critical');
    expect(getCountdownSeverity(-1)).toBe('critical');
  });
});
