import { getCountdownSeverity, msToDisplaySeconds } from '../../src/client/utils/countdown';

describe('countdown utils', () => {
  describe('getCountdownSeverity', () => {
    it('returns null for non-numeric values', () => {
      expect(getCountdownSeverity(null)).toBeNull();
      expect(getCountdownSeverity(undefined)).toBeNull();
      expect(getCountdownSeverity(NaN)).toBeNull();
    });

    it('classifies time remaining into normal/warning/critical buckets', () => {
      expect(getCountdownSeverity(15_000)).toBe('normal');
      expect(getCountdownSeverity(10_000)).toBe('warning');
      expect(getCountdownSeverity(4_000)).toBe('warning');
      expect(getCountdownSeverity(3_000)).toBe('critical');
      expect(getCountdownSeverity(0)).toBe('critical');
      expect(getCountdownSeverity(-500)).toBe('critical');
    });
  });

  describe('msToDisplaySeconds', () => {
    it('returns null for nullish or non-finite values', () => {
      expect(msToDisplaySeconds(null)).toBeNull();
      expect(msToDisplaySeconds(undefined)).toBeNull();
      expect(msToDisplaySeconds(NaN)).toBeNull();
      expect(msToDisplaySeconds(Infinity)).toBeNull();
    });

    it('rounds up positive values and clamps negative to zero', () => {
      expect(msToDisplaySeconds(-200)).toBe(0);
      expect(msToDisplaySeconds(0)).toBe(0);
      expect(msToDisplaySeconds(1)).toBe(1);
      expect(msToDisplaySeconds(999)).toBe(1);
      expect(msToDisplaySeconds(1000)).toBe(1);
      expect(msToDisplaySeconds(1001)).toBe(2);
      expect(msToDisplaySeconds(1999)).toBe(2);
      expect(msToDisplaySeconds(2500)).toBe(3);
    });
  });
});
