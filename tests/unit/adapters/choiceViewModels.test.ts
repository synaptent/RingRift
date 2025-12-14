import {
  getChoiceViewModelForType,
  getChoiceViewModel,
} from '../../../src/client/adapters/choiceViewModels';
import type { PlayerChoiceType } from '../../../src/shared/types/game';

describe('choiceViewModels adapter', () => {
  const ALL_TYPES: PlayerChoiceType[] = [
    'line_order',
    'line_reward_option',
    'ring_elimination',
    'region_order',
    'capture_direction',
  ];

  it('returns a view model for every known PlayerChoiceType', () => {
    for (const type of ALL_TYPES) {
      const vm = getChoiceViewModelForType(type);

      expect(vm.type).toBe(type);
      expect(vm.kind).toBeDefined();
      expect(vm.copy.title).toBeTruthy();
      expect(vm.copy.shortLabel).toBeTruthy();
      expect(vm.timeout.showCountdown).toBe(true);
    }
  });

  it('provides stable, human-readable titles for each decision type', () => {
    const expectations: Array<{ type: PlayerChoiceType; title: RegExp; shortLabel: RegExp }> = [
      {
        type: 'line_order',
        title: /multiple lines formed/i,
        shortLabel: /line order/i,
      },
      {
        type: 'line_reward_option',
        title: /line scored.*choose your reward/i,
        shortLabel: /line reward/i,
      },
      {
        type: 'ring_elimination',
        title: /remove a ring/i,
        shortLabel: /ring elimination/i,
      },
      {
        type: 'region_order',
        title: /territory captured/i,
        shortLabel: /territory region/i,
      },
      {
        type: 'capture_direction',
        title: /chain capture.*keep jumping/i,
        shortLabel: /capture direction/i,
      },
    ];

    for (const { type, title, shortLabel } of expectations) {
      const vm = getChoiceViewModelForType(type);
      expect(vm.copy.title).toMatch(title);
      expect(vm.copy.shortLabel).toMatch(shortLabel);
    }
  });

  it('formats spectator labels using the acting player name', () => {
    const actingPlayerName = 'Aria';

    for (const type of ALL_TYPES) {
      const vm = getChoiceViewModelForType(type);
      const spectatorText = vm.copy.spectatorLabel({ actingPlayerName });

      expect(spectatorText).toContain(actingPlayerName);
      expect(typeof spectatorText).toBe('string');
      expect(spectatorText.length).toBeGreaterThan(0);
    }
  });

  it('falls back to a generic decision view model for unknown types', () => {
    const unknownType = 'debug_only_choice' as PlayerChoiceType;
    const vm = getChoiceViewModelForType(unknownType);

    expect(vm.type).toBe(unknownType);
    expect(vm.kind).toBe('other');
    expect(vm.copy.title).toMatch(/decision required/i);
    expect(vm.copy.shortLabel).toMatch(/decision/i);
  });

  it('getChoiceViewModel convenience helper delegates to getChoiceViewModelForType', () => {
    const choice = { type: 'line_order' as PlayerChoiceType };

    const direct = getChoiceViewModelForType('line_order');
    const viaChoice = getChoiceViewModel(choice);

    expect(viaChoice).toEqual(direct);
  });
});
