import { getEffectiveLineLengthThreshold } from '../../src/shared/engine';
import { BOARD_CONFIGS } from '../../src/shared/types/game';

describe('rulesConfig – getEffectiveLineLengthThreshold', () => {
  it('returns BOARD_CONFIGS lineLength for non-square8 boards', () => {
    const square19Base = BOARD_CONFIGS.square19.lineLength;
    const hexBase = BOARD_CONFIGS.hexagonal.lineLength;

    expect(getEffectiveLineLengthThreshold('square19', 2)).toBe(square19Base);
    expect(getEffectiveLineLengthThreshold('square19', 3)).toBe(square19Base);
    expect(getEffectiveLineLengthThreshold('square19', 4)).toBe(square19Base);

    expect(getEffectiveLineLengthThreshold('hexagonal', 2)).toBe(hexBase);
    expect(getEffectiveLineLengthThreshold('hexagonal', 3)).toBe(hexBase);
    expect(getEffectiveLineLengthThreshold('hexagonal', 4)).toBe(hexBase);
  });

  it('uses BOARD_CONFIGS lineLength for square8 except in 2-player games', () => {
    const base = BOARD_CONFIGS.square8.lineLength;

    // 3p/4p square8 use the base threshold (e.g., 3-in-a-row).
    expect(getEffectiveLineLengthThreshold('square8', 3)).toBe(base);
    expect(getEffectiveLineLengthThreshold('square8', 4)).toBe(base);
  });

  it('elevates square8 threshold to 4-in-a-row for 2-player games', () => {
    const base = BOARD_CONFIGS.square8.lineLength;

    // Sanity: base config is the lower threshold (3-in-a-row).
    expect(base).toBeLessThanOrEqual(4);

    const effective = getEffectiveLineLengthThreshold('square8', 2);
    expect(effective).toBe(4);
    expect(effective).toBeGreaterThanOrEqual(base);
  });
});

