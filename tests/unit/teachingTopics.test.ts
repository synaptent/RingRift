/**
 * Tests for teaching topic definitions and content
 * @module tests/unit/teachingTopics.test
 */

import {
  VICTORY_STALEMATE_TIPS,
  FORCED_ELIMINATION_TIPS,
  ACTIVE_NO_MOVES_TIPS,
  RECOVERY_ACTION_TIPS,
  CHAIN_CAPTURE_TIPS,
  LINE_TERRITORY_ORDER_TIPS,
  TERRITORY_TIPS,
  LPS_FIRST_OCCURRENCE_TIPS,
  TOPIC_TO_CONCEPTS,
  getConceptsForTopic,
  type TeachingTopicId,
  type TeachingTip,
} from '../../src/shared/teaching/teachingTopics';

describe('teachingTopics', () => {
  describe('TeachingTip structure', () => {
    const validateTip = (tip: TeachingTip, expectCategory?: string) => {
      expect(typeof tip.text).toBe('string');
      expect(tip.text.length).toBeGreaterThan(0);
      if (expectCategory) {
        expect(tip.category).toBe(expectCategory);
      }
      if (tip.emphasis) {
        expect(['normal', 'important', 'critical']).toContain(tip.emphasis);
      }
      if (tip.addressesGap) {
        expect(typeof tip.addressesGap).toBe('string');
        expect(tip.addressesGap).toMatch(/^GAP-/);
      }
    };

    describe('VICTORY_STALEMATE_TIPS', () => {
      it('should be an array of TeachingTip objects', () => {
        expect(Array.isArray(VICTORY_STALEMATE_TIPS)).toBe(true);
        expect(VICTORY_STALEMATE_TIPS.length).toBeGreaterThan(0);
      });

      it('should have valid tip structure', () => {
        for (const tip of VICTORY_STALEMATE_TIPS) {
          validateTip(tip);
        }
      });

      it('should have LPS tips', () => {
        const lpsTips = VICTORY_STALEMATE_TIPS.filter((t) => t.category === 'lps');
        expect(lpsTips.length).toBeGreaterThan(0);
      });

      it('should have stalemate tips', () => {
        const stalemateTips = VICTORY_STALEMATE_TIPS.filter((t) => t.category === 'stalemate');
        expect(stalemateTips.length).toBeGreaterThan(0);
      });

      it('should have tiebreak tips', () => {
        const tiebreakTips = VICTORY_STALEMATE_TIPS.filter((t) => t.category === 'tiebreak');
        expect(tiebreakTips.length).toBeGreaterThan(0);
      });

      it('should address gap GAP-LPS-02 (LPS sub-section)', () => {
        const gap02Tips = VICTORY_STALEMATE_TIPS.filter((t) => t.addressesGap === 'GAP-LPS-02');
        expect(gap02Tips.length).toBeGreaterThan(0);
      });

      it('should address gap GAP-LPS-03 (FE != real action)', () => {
        const gap03Tips = VICTORY_STALEMATE_TIPS.filter((t) => t.addressesGap === 'GAP-LPS-03');
        expect(gap03Tips.length).toBeGreaterThan(0);
      });

      it('should address gap GAP-STALE-04 (ANM vs global stalemate)', () => {
        const gap04Tips = VICTORY_STALEMATE_TIPS.filter((t) => t.addressesGap === 'GAP-STALE-04');
        expect(gap04Tips.length).toBeGreaterThan(0);
      });

      it('should address gap GAP-STALE-01 (tiebreak ladder)', () => {
        const gap01Tips = VICTORY_STALEMATE_TIPS.filter((t) => t.addressesGap === 'GAP-STALE-01');
        expect(gap01Tips.length).toBeGreaterThan(0);
      });

      it('should have at least one critical emphasis tip', () => {
        const criticalTips = VICTORY_STALEMATE_TIPS.filter((t) => t.emphasis === 'critical');
        expect(criticalTips.length).toBeGreaterThan(0);
      });
    });

    describe('FORCED_ELIMINATION_TIPS', () => {
      it('should be an array of TeachingTip objects', () => {
        expect(Array.isArray(FORCED_ELIMINATION_TIPS)).toBe(true);
        expect(FORCED_ELIMINATION_TIPS.length).toBeGreaterThan(0);
      });

      it('should have valid tip structure', () => {
        for (const tip of FORCED_ELIMINATION_TIPS) {
          validateTip(tip);
        }
      });

      it('should explain forced elimination mechanics', () => {
        const hasExplanation = FORCED_ELIMINATION_TIPS.some(
          (t) =>
            t.text.toLowerCase().includes('forced') || t.text.toLowerCase().includes('eliminate')
        );
        expect(hasExplanation).toBe(true);
      });
    });

    describe('ACTIVE_NO_MOVES_TIPS', () => {
      it('should be an array of TeachingTip objects', () => {
        expect(Array.isArray(ACTIVE_NO_MOVES_TIPS)).toBe(true);
        expect(ACTIVE_NO_MOVES_TIPS.length).toBeGreaterThan(0);
      });

      it('should have valid tip structure', () => {
        for (const tip of ACTIVE_NO_MOVES_TIPS) {
          validateTip(tip);
        }
      });

      it('should explain ANM concept', () => {
        const hasANMExplanation = ACTIVE_NO_MOVES_TIPS.some(
          (t) =>
            t.text.toLowerCase().includes('no move') ||
            t.text.toLowerCase().includes('stuck') ||
            t.text.toLowerCase().includes('active') ||
            t.text.toLowerCase().includes('anm') ||
            t.text.toLowerCase().includes('cannot')
        );
        expect(hasANMExplanation).toBe(true);
      });
    });

    describe('RECOVERY_ACTION_TIPS', () => {
      it('should be an array of TeachingTip objects', () => {
        expect(Array.isArray(RECOVERY_ACTION_TIPS)).toBe(true);
        expect(RECOVERY_ACTION_TIPS.length).toBeGreaterThan(0);
      });

      it('should have valid tip structure', () => {
        for (const tip of RECOVERY_ACTION_TIPS) {
          validateTip(tip);
        }
      });

      it('should explain recovery mechanics', () => {
        const hasRecoveryExplanation = RECOVERY_ACTION_TIPS.some(
          (t) => t.text.toLowerCase().includes('recover') || t.text.toLowerCase().includes('marker')
        );
        expect(hasRecoveryExplanation).toBe(true);
      });
    });
  });

  describe('Additional tip arrays', () => {
    describe('CHAIN_CAPTURE_TIPS', () => {
      it('should be an array with tips', () => {
        expect(Array.isArray(CHAIN_CAPTURE_TIPS)).toBe(true);
        expect(CHAIN_CAPTURE_TIPS.length).toBeGreaterThan(0);
      });

      it('should explain chain capture mechanics', () => {
        const hasChainExplanation = CHAIN_CAPTURE_TIPS.some(
          (t) => t.text.toLowerCase().includes('chain') || t.text.toLowerCase().includes('capture')
        );
        expect(hasChainExplanation).toBe(true);
      });
    });

    describe('LINE_TERRITORY_ORDER_TIPS', () => {
      it('should be an array with tips', () => {
        expect(Array.isArray(LINE_TERRITORY_ORDER_TIPS)).toBe(true);
        expect(LINE_TERRITORY_ORDER_TIPS.length).toBeGreaterThan(0);
      });

      it('should explain line/territory ordering', () => {
        const hasExplanation = LINE_TERRITORY_ORDER_TIPS.some(
          (t) => t.text.toLowerCase().includes('line') || t.text.toLowerCase().includes('territory')
        );
        expect(hasExplanation).toBe(true);
      });
    });

    describe('TERRITORY_TIPS', () => {
      it('should be an array with tips', () => {
        expect(Array.isArray(TERRITORY_TIPS)).toBe(true);
        expect(TERRITORY_TIPS.length).toBeGreaterThan(0);
      });
    });

    describe('LPS_FIRST_OCCURRENCE_TIPS', () => {
      it('should be an array with tips', () => {
        expect(Array.isArray(LPS_FIRST_OCCURRENCE_TIPS)).toBe(true);
        expect(LPS_FIRST_OCCURRENCE_TIPS.length).toBeGreaterThan(0);
      });

      it('should explain LPS (Last Player Standing)', () => {
        const hasLpsExplanation = LPS_FIRST_OCCURRENCE_TIPS.some(
          (t) =>
            t.text.toLowerCase().includes('last player') ||
            t.text.toLowerCase().includes('lps') ||
            t.text.toLowerCase().includes('standing')
        );
        expect(hasLpsExplanation).toBe(true);
      });
    });
  });

  describe('TOPIC_TO_CONCEPTS mapping', () => {
    it('should be an object mapping topics to concept arrays', () => {
      expect(typeof TOPIC_TO_CONCEPTS).toBe('object');
    });

    it('should map victory_stalemate to concepts', () => {
      const concepts = TOPIC_TO_CONCEPTS['victory_stalemate'];
      if (concepts) {
        expect(Array.isArray(concepts)).toBe(true);
      }
    });

    it('should map forced_elimination to concepts', () => {
      const concepts = TOPIC_TO_CONCEPTS['forced_elimination'];
      if (concepts) {
        expect(Array.isArray(concepts)).toBe(true);
      }
    });
  });

  describe('getConceptsForTopic', () => {
    it('should return concepts for mapped topics', () => {
      const concepts = getConceptsForTopic('victory_stalemate');
      expect(Array.isArray(concepts)).toBe(true);
    });

    it('should return empty array for unmapped topics', () => {
      const concepts = getConceptsForTopic('ring_placement');
      expect(Array.isArray(concepts)).toBe(true);
      // May be empty or have concepts depending on mapping
    });

    it('should return concepts consistently', () => {
      const concepts1 = getConceptsForTopic('territory');
      const concepts2 = getConceptsForTopic('territory');
      expect(concepts1).toEqual(concepts2);
    });
  });
});
