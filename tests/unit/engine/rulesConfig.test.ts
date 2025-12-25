/**
 * Test suite for src/shared/engine/rulesConfig.ts
 *
 * Tests rules configuration functions for line length thresholds
 * and rings per player calculations.
 */

import {
  getEffectiveLineLengthThreshold,
  getEffectiveRingsPerPlayer,
} from '../../../src/shared/engine/rulesConfig';
import { BOARD_CONFIGS, type BoardType, type RulesOptions } from '../../../src/shared/types/game';

describe('rulesConfig', () => {
  describe('getEffectiveLineLengthThreshold', () => {
    describe('square8 board', () => {
      it('should return 4 for 2-player games (RR-CANON-R120)', () => {
        expect(getEffectiveLineLengthThreshold('square8', 2)).toBe(4);
      });

      it('should return 3 for 3-player games (RR-CANON-R120)', () => {
        expect(getEffectiveLineLengthThreshold('square8', 3)).toBe(3);
      });

      it('should return 3 for 4-player games (RR-CANON-R120)', () => {
        expect(getEffectiveLineLengthThreshold('square8', 4)).toBe(3);
      });
    });

    describe('hex8 board', () => {
      it('should return 4 for 2-player games', () => {
        expect(getEffectiveLineLengthThreshold('hex8', 2)).toBe(4);
      });

      it('should return 3 for 3-player games', () => {
        expect(getEffectiveLineLengthThreshold('hex8', 3)).toBe(3);
      });

      it('should return 3 for 4-player games', () => {
        expect(getEffectiveLineLengthThreshold('hex8', 4)).toBe(3);
      });
    });

    describe('square19 board', () => {
      it('should return 4 for all player counts', () => {
        expect(getEffectiveLineLengthThreshold('square19', 2)).toBe(4);
        expect(getEffectiveLineLengthThreshold('square19', 3)).toBe(4);
        expect(getEffectiveLineLengthThreshold('square19', 4)).toBe(4);
      });
    });

    describe('hexagonal board', () => {
      it('should return 4 for all player counts', () => {
        expect(getEffectiveLineLengthThreshold('hexagonal', 2)).toBe(4);
        expect(getEffectiveLineLengthThreshold('hexagonal', 3)).toBe(4);
        expect(getEffectiveLineLengthThreshold('hexagonal', 4)).toBe(4);
      });
    });

    it('should accept unused rulesOptions parameter', () => {
      const rulesOptions: RulesOptions = { swapRuleEnabled: true };
      expect(getEffectiveLineLengthThreshold('square8', 2, rulesOptions)).toBe(4);
    });

    it('should handle undefined rulesOptions', () => {
      expect(getEffectiveLineLengthThreshold('square8', 2, undefined)).toBe(4);
    });
  });

  describe('getEffectiveRingsPerPlayer', () => {
    describe('default values from BOARD_CONFIGS', () => {
      it('should return default for square8', () => {
        expect(getEffectiveRingsPerPlayer('square8')).toBe(BOARD_CONFIGS.square8.ringsPerPlayer);
      });

      it('should return default for square19', () => {
        expect(getEffectiveRingsPerPlayer('square19')).toBe(BOARD_CONFIGS.square19.ringsPerPlayer);
      });

      it('should return default for hexagonal', () => {
        expect(getEffectiveRingsPerPlayer('hexagonal')).toBe(
          BOARD_CONFIGS.hexagonal.ringsPerPlayer
        );
      });

      it('should return default for hex8', () => {
        expect(getEffectiveRingsPerPlayer('hex8')).toBe(BOARD_CONFIGS.hex8.ringsPerPlayer);
      });
    });

    describe('rulesOptions override', () => {
      it('should use rulesOptions.ringsPerPlayer when provided', () => {
        const rulesOptions: RulesOptions = { ringsPerPlayer: 50 };
        expect(getEffectiveRingsPerPlayer('square8', rulesOptions)).toBe(50);
      });

      it('should fall back to default when rulesOptions has no ringsPerPlayer', () => {
        const rulesOptions: RulesOptions = { swapRuleEnabled: true };
        expect(getEffectiveRingsPerPlayer('square8', rulesOptions)).toBe(
          BOARD_CONFIGS.square8.ringsPerPlayer
        );
      });

      it('should use override for any board type', () => {
        const rulesOptions: RulesOptions = { ringsPerPlayer: 100 };
        expect(getEffectiveRingsPerPlayer('hexagonal', rulesOptions)).toBe(100);
        expect(getEffectiveRingsPerPlayer('square19', rulesOptions)).toBe(100);
      });
    });

    describe('edge cases', () => {
      it('should handle undefined rulesOptions', () => {
        expect(getEffectiveRingsPerPlayer('square8', undefined)).toBe(
          BOARD_CONFIGS.square8.ringsPerPlayer
        );
      });

      it('should handle empty rulesOptions', () => {
        expect(getEffectiveRingsPerPlayer('square8', {})).toBe(
          BOARD_CONFIGS.square8.ringsPerPlayer
        );
      });
    });
  });

  describe('BOARD_CONFIGS consistency', () => {
    it('should have valid line lengths configured for all boards', () => {
      // Each board type has a lineLength configured
      const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal', 'hex8'];
      boardTypes.forEach((bt) => {
        expect(BOARD_CONFIGS[bt].lineLength).toBeGreaterThanOrEqual(3);
        expect(BOARD_CONFIGS[bt].lineLength).toBeLessThanOrEqual(5);
      });
    });

    it('should have ringsPerPlayer configured for all board types', () => {
      const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal', 'hex8'];
      boardTypes.forEach((bt) => {
        expect(BOARD_CONFIGS[bt].ringsPerPlayer).toBeGreaterThan(0);
      });
    });
  });
});
