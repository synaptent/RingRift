import {
  DIFFICULTY_DESCRIPTORS,
  getDifficultyDescriptor,
} from '../../src/client/utils/difficultyUx';

describe('difficultyUx descriptors', () => {
  it('defines descriptors for difficulties 1 through 8 with non-empty fields', () => {
    for (let id = 1; id <= 8; id += 1) {
      const descriptor = getDifficultyDescriptor(id);
      expect(descriptor).toBeDefined();

      expect(typeof descriptor!.name).toBe('string');
      expect(descriptor!.name.trim().length).toBeGreaterThan(0);

      expect(typeof descriptor!.shortDescription).toBe('string');
      expect(descriptor!.shortDescription.trim().length).toBeGreaterThan(0);

      expect(typeof descriptor!.detailedDescription).toBe('string');
      expect(descriptor!.detailedDescription.trim().length).toBeGreaterThan(0);

      expect(typeof descriptor!.recommendedAudience).toBe('string');
      expect(descriptor!.recommendedAudience.trim().length).toBeGreaterThan(0);
    }
  });

  it('marks core calibration tiers D2/D4/D6/D8 with appropriate difficulty anchors', () => {
    const d2 = getDifficultyDescriptor(2);
    const d4 = getDifficultyDescriptor(4);
    const d6 = getDifficultyDescriptor(6);
    const d8 = getDifficultyDescriptor(8);

    expect(d2).toBeDefined();
    expect(d4).toBeDefined();
    expect(d6).toBeDefined();
    expect(d8).toBeDefined();

    const d2Text = `${d2!.name} ${d2!.shortDescription}`.toLowerCase();
    expect(d2Text).toContain('learner');
    expect(d2Text).toContain('casual');

    const d4Text = `${d4!.name} ${d4!.shortDescription} ${d4!.recommendedAudience}`.toLowerCase();
    expect(d4Text).toContain('challenging');
    expect(d4Text).toContain('intermediate');

    const d6Text = `${d6!.name} ${d6!.shortDescription}`.toLowerCase();
    expect(d6Text).toContain('advanced');
    expect(d6Text).toContain('strong');

    const d8Text = `${d8!.name} ${d8!.shortDescription}`.toLowerCase();
    expect(d8Text).toContain('expert');
    expect(d8Text).toContain('strong');
  });

  it('flags experimental and unrated status for high-end tiers D9 and D10', () => {
    const d9 = getDifficultyDescriptor(9);
    const d10 = getDifficultyDescriptor(10);

    expect(d9).toBeDefined();
    expect(d10).toBeDefined();

    expect(typeof d9!.notes).toBe('string');
    expect(typeof d10!.notes).toBe('string');

    const d9Notes = d9!.notes!.toLowerCase();
    const d10Notes = d10!.notes!.toLowerCase();

    // D9 is explicitly described as experimental / unrated.
    expect(d9Notes).toContain('experimental');
    expect(d9Notes).toContain('unrated');

    // D10 is explicitly unrated and outside the calibration guide.
    expect(d10Notes).toContain('unrated');
  });

  it('keeps DIFFICULTY_DESCRIPTORS ordered and containing at least 10 entries', () => {
    expect(DIFFICULTY_DESCRIPTORS.length).toBeGreaterThanOrEqual(10);

    // Ensure ids are unique and strictly increasing so lookups remain stable.
    const ids = DIFFICULTY_DESCRIPTORS.map((d) => d.id);
    const sorted = [...ids].sort((a, b) => a - b);
    expect(ids).toEqual(sorted);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
