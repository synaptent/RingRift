import '@testing-library/jest-dom';
import type { GameResult } from '../../src/shared/types/game';
import type {
  LoadableScenario,
  ScenarioRulesConcept,
} from '../../src/client/sandbox/scenarioTypes';
import {
  logSandboxScenarioLoaded,
  logSandboxScenarioCompleted,
} from '../../src/client/sandbox/sandboxRulesUxTelemetry';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';

jest.mock('../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../src/client/utils/rulesUxTelemetry');
  return {
    __esModule: true,
    ...actual,
    // Replace logRulesUxEvent with a Jest mock so we can assert on the
    // final telemetry envelope without performing real network calls.
    logRulesUxEvent: jest.fn(),
  };
});

const mockLogRulesUxEvent = rulesUxTelemetry.logRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logRulesUxEvent
>;

function createTeachingScenario(overrides: Partial<LoadableScenario> = {}): LoadableScenario {
  const base: LoadableScenario = {
    id: 'teaching.fe_loop.step_1',
    name: 'Forced Elimination – intro',
    description: 'Intro teaching step for Active–No–Moves / Forced Elimination.',
    category: 'learning',
    tags: [],
    boardType: 'square8',
    playerCount: 2,
    createdAt: new Date().toISOString(),
    source: 'curated',
    state: {} as any,
    // Bridge from curated sandbox metadata to shared teaching metadata:
    // both use the anm_forced_elimination concept for this flow.
    rulesConcept: 'anm_forced_elimination' as ScenarioRulesConcept,
  };

  return { ...base, ...overrides };
}

describe('sandboxRulesUxTelemetry – sandbox scenario events', () => {
  beforeEach(() => {
    mockLogRulesUxEvent.mockReset();
  });

  it('emits sandbox_scenario_loaded with RulesUxContext mapped from teaching metadata', async () => {
    const scenario = createTeachingScenario();

    await logSandboxScenarioLoaded(scenario);

    expect(mockLogRulesUxEvent).toHaveBeenCalledTimes(1);
    const event = mockLogRulesUxEvent.mock.calls[0][0];

    expect(event).toMatchObject({
      type: 'sandbox_scenario_loaded',
      source: 'sandbox',
      boardType: 'square8',
      numPlayers: 2,
      scenarioId: scenario.id,
      rulesConcept: scenario.rulesConcept,
      isSandbox: true,
    });
    // For the initial FE loop teaching flow, the telemetry context should
    // resolve to the canonical anm_forced_elimination RulesUxContext.
    expect(event.rulesContext).toBe('anm_forced_elimination');
  });

  it('emits sandbox_scenario_completed with success payload for winning completions', async () => {
    const scenario = createTeachingScenario();
    const victoryReason: GameResult['reason'] = 'ring_elimination';

    await logSandboxScenarioCompleted({ scenario, victoryReason });

    expect(mockLogRulesUxEvent).toHaveBeenCalledTimes(1);
    const event = mockLogRulesUxEvent.mock.calls[0][0];

    expect(event).toMatchObject({
      type: 'sandbox_scenario_completed',
      source: 'sandbox',
      boardType: 'square8',
      numPlayers: 2,
      scenarioId: scenario.id,
      rulesConcept: scenario.rulesConcept,
      isSandbox: true,
    });
    expect(event.rulesContext).toBe('anm_forced_elimination');
    expect(event.payload).toEqual(
      expect.objectContaining({
        success: true,
        victoryReason: 'ring_elimination',
      })
    );
  });

  it('does not emit telemetry for non-curated scenarios', async () => {
    const curated = createTeachingScenario();
    const custom = createTeachingScenario({
      id: 'custom-scenario',
      source: 'custom',
    });

    await logSandboxScenarioLoaded(curated);
    await logSandboxScenarioLoaded(custom);
    await logSandboxScenarioCompleted({ scenario: curated, victoryReason: 'ring_elimination' });
    await logSandboxScenarioCompleted({ scenario: custom, victoryReason: 'ring_elimination' });

    // Only the curated scenario should produce telemetry events.
    expect(mockLogRulesUxEvent).toHaveBeenCalledTimes(2);

    const types = mockLogRulesUxEvent.mock.calls.map(([arg]) => (arg as any).type).sort();
    expect(types).toEqual(['sandbox_scenario_completed', 'sandbox_scenario_loaded'].sort());
  });
});
