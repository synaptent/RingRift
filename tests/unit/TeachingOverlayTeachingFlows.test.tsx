import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TeachingOverlay } from '../../src/client/components/TeachingOverlay';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';
import { TEACHING_TOPICS_COPY } from '../../src/client/utils/rulesUxTelemetry';

jest.mock('../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../src/client/utils/rulesUxTelemetry');
  return {
    __esModule: true,
    ...actual,
    logRulesUxEvent: jest.fn(),
  };
});

const mockLogRulesUxEvent = rulesUxTelemetry.logRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logRulesUxEvent
>;

describe('TeachingOverlay teaching flows – step navigation & telemetry', () => {
  beforeEach(() => {
    mockLogRulesUxEvent.mockReset();
  });

  it('renders canonical headings for key weird-state topics', () => {
    const cases: Array<{ topic: any; expectedHeading: string }> = [
      { topic: 'active_no_moves', expectedHeading: TEACHING_TOPICS_COPY.active_no_moves.heading },
      {
        topic: 'forced_elimination',
        expectedHeading: TEACHING_TOPICS_COPY.forced_elimination.heading,
      },
      {
        topic: 'victory_stalemate',
        expectedHeading: TEACHING_TOPICS_COPY.victory_stalemate.heading,
      },
      { topic: 'territory', expectedHeading: TEACHING_TOPICS_COPY.territory.heading },
    ];

    for (const { topic, expectedHeading } of cases) {
      render(<TeachingOverlay topic={topic} isOpen={true} onClose={() => {}} position="center" />);
      expect(
        screen.getByRole('heading', {
          name: expectedHeading,
        })
      ).toBeInTheDocument();
    }
  });

  it('emits teaching_step_started when a related teaching step is selected', async () => {
    render(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const stepButtons = await screen.findAllByTestId('teaching-related-step');
    expect(stepButtons.length).toBeGreaterThan(0);

    fireEvent.click(stepButtons[0]);

    await waitFor(() => {
      expect(mockLogRulesUxEvent).toHaveBeenCalled();
    });

    const startedEvents = mockLogRulesUxEvent.mock.calls
      .map(([arg]) => arg as any)
      .filter((event) => event.type === 'teaching_step_started');
    expect(startedEvents.length).toBeGreaterThanOrEqual(1);
    const event = startedEvents[0];

    expect(event.source).toBe('teaching_overlay');
    expect(event.rulesConcept).toBe('anm_forced_elimination');
    expect(event.boardType).toBe('square8');
    expect(event.numPlayers).toBe(2);
    expect(typeof event.teachingFlowId).toBe('string');
    expect(event.teachingFlowId.length).toBeGreaterThan(0);
    expect(event.scenarioId).toMatch(/^teaching\.fe_loop\.step_/);
    expect(event.payload).toEqual(
      expect.objectContaining({
        flowId: 'fe_loop_intro',
        stepIndex: expect.any(Number),
        topic: 'forced_elimination',
      })
    );
  });

  it('emits teaching_step_completed when the step is marked as understood', async () => {
    render(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const stepButtons = await screen.findAllByTestId('teaching-related-step');
    fireEvent.click(stepButtons[0]);

    const completeButton = await screen.findByRole('button', { name: /mark as understood/i });
    fireEvent.click(completeButton);

    await waitFor(() => {
      expect(mockLogRulesUxEvent).toHaveBeenCalled();
    });

    const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
    const started = events.filter((e) => e.type === 'teaching_step_started');
    const completed = events.filter((e) => e.type === 'teaching_step_completed');
    expect(started.length).toBeGreaterThanOrEqual(1);
    expect(completed.length).toBeGreaterThanOrEqual(1);
    const startedEvent = started[started.length - 1];
    const completedEvent = completed[completed.length - 1];

    expect(completedEvent.source).toBe('teaching_overlay');
    expect(completedEvent.rulesConcept).toBe('anm_forced_elimination');
    expect(completedEvent.boardType).toBe('square8');
    expect(completedEvent.numPlayers).toBe(2);
    expect(completedEvent.scenarioId).toEqual(startedEvent.scenarioId);
    expect(completedEvent.teachingFlowId).toEqual(startedEvent.teachingFlowId);
    expect(completedEvent.payload).toEqual(
      expect.objectContaining({
        flowId: 'fe_loop_intro',
        stepIndex: startedEvent.payload.stepIndex,
        topic: 'forced_elimination',
        completionAction: 'mark_understood',
      })
    );
  });
  it('auto-selects a step when opened with a matching weird-state context and emits teaching_step_started', async () => {
    const weirdStateOverlayContext = {
      reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
      rulesContext: 'anm_forced_elimination',
      weirdStateType: 'active-no-moves-movement',
      boardType: 'square8',
      numPlayers: 2,
      isRanked: false,
      isSandbox: true,
      overlaySessionId: 'overlay-session-1',
    } as const;

    render(
      <TeachingOverlay
        topic="active_no_moves"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={weirdStateOverlayContext as any}
      />
    );

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls
        .map(([arg]) => arg as any)
        .filter((e) => e.type === 'teaching_step_started');
      expect(events.length).toBeGreaterThanOrEqual(1);
    });

    const events = mockLogRulesUxEvent.mock.calls
      .map(([arg]) => arg as any)
      .filter((e) => e.type === 'teaching_step_started');
    const event = events[events.length - 1];
    expect(event.overlaySessionId).toBe(weirdStateOverlayContext.overlaySessionId);
    expect(event.payload).toEqual(
      expect.objectContaining({
        startedAutomatically: true,
        topic: 'active_no_moves',
      })
    );
  });

  it('auto-selects a structural stalemate teaching step for STRUCTURAL_STALEMATE_TIEBREAK weird-state context', async () => {
    const weirdStateOverlayContext = {
      reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
      rulesContext: 'structural_stalemate',
      weirdStateType: 'victory-structural-stalemate',
      boardType: 'square8',
      numPlayers: 2,
      isRanked: false,
      isSandbox: true,
      overlaySessionId: 'overlay-session-structural-stalemate',
    } as const;

    render(
      <TeachingOverlay
        topic="victory_stalemate"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={weirdStateOverlayContext as any}
      />
    );

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls
        .map(([arg]) => arg as any)
        .filter((e) => e.type === 'teaching_step_started');
      expect(events.length).toBeGreaterThanOrEqual(1);
    });

    const events = mockLogRulesUxEvent.mock.calls
      .map(([arg]) => arg as any)
      .filter((e) => e.type === 'teaching_step_started');
    const event = events[events.length - 1];

    expect(event.rulesConcept).toBe('structural_stalemate');
    expect(event.scenarioId).toMatch(/^teaching\.structural_stalemate\.step_/);
    expect(event.overlaySessionId).toBe(weirdStateOverlayContext.overlaySessionId);
    expect(event.payload).toEqual(
      expect.objectContaining({
        startedAutomatically: true,
        topic: 'victory_stalemate',
      })
    );
  });

  it('auto-selects a Last Player Standing teaching step for LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS weird-state context', async () => {
    const weirdStateOverlayContext = {
      reasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
      rulesContext: 'last_player_standing',
      weirdStateType: 'victory-last-player-standing',
      boardType: 'square8',
      numPlayers: 3,
      isRanked: false,
      isSandbox: true,
      overlaySessionId: 'overlay-session-lps',
    } as const;

    render(
      <TeachingOverlay
        topic="victory_stalemate"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={weirdStateOverlayContext as any}
      />
    );

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls
        .map(([arg]) => arg as any)
        .filter((e) => e.type === 'teaching_step_started');
      expect(events.length).toBeGreaterThanOrEqual(1);
    });

    const events = mockLogRulesUxEvent.mock.calls
      .map(([arg]) => arg as any)
      .filter((e) => e.type === 'teaching_step_started');
    const event = events[events.length - 1];

    expect(event.rulesConcept).toBe('last_player_standing');
    expect(event.scenarioId).toMatch(/^teaching\.lps\.step_/);
    expect(event.overlaySessionId).toBe(weirdStateOverlayContext.overlaySessionId);
    expect(event.payload).toEqual(
      expect.objectContaining({
        startedAutomatically: true,
        topic: 'victory_stalemate',
      })
    );
  });

  it('auto-selects a territory mini-region teaching step for ANM_TERRITORY_NO_ACTIONS weird-state context', async () => {
    const weirdStateOverlayContext = {
      reasonCode: 'ANM_TERRITORY_NO_ACTIONS',
      rulesContext: 'territory_mini_region',
      weirdStateType: 'territory-no-actions-mini-region',
      boardType: 'square8',
      numPlayers: 2,
      isRanked: false,
      isSandbox: true,
      overlaySessionId: 'overlay-session-territory-anm',
    } as const;

    render(
      <TeachingOverlay
        topic="territory"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={weirdStateOverlayContext as any}
      />
    );

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls
        .map(([arg]) => arg as any)
        .filter((e) => e.type === 'teaching_step_started');
      expect(events.length).toBeGreaterThanOrEqual(1);
    });

    const events = mockLogRulesUxEvent.mock.calls
      .map(([arg]) => arg as any)
      .filter((e) => e.type === 'teaching_step_started');
    const event = events[events.length - 1];

    expect(event.rulesConcept).toBe('territory_mini_region');
    expect(event.scenarioId).toMatch(/^teaching\.mini_region\.step_/);
    expect(event.overlaySessionId).toBe(weirdStateOverlayContext.overlaySessionId);
    expect(event.payload).toEqual(
      expect.objectContaining({
        startedAutomatically: true,
        topic: 'territory',
      })
    );
  });

  it('surfaces the multi-turn forced-elimination loop step and emits telemetry with the correct flow metadata', async () => {
    render(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const stepButtons = await screen.findAllByTestId('teaching-related-step');

    // Click all related steps once so any fe_loop_intro step with the highest
    // index (our multi-turn FE loop) has a chance to emit telemetry. We avoid
    // depending on the exact rendered label text here and instead assert via
    // the telemetry payload.
    stepButtons.forEach((button) => {
      fireEvent.click(button);
    });

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls
        .map(([arg]) => arg as any)
        .filter((e) => e.type === 'teaching_step_started');
      expect(events.length).toBeGreaterThanOrEqual(1);
    });

    const events = mockLogRulesUxEvent.mock.calls
      .map(([arg]) => arg as any)
      .filter((e) => e.type === 'teaching_step_started');

    const feLoopStep4Events = events.filter(
      (event) =>
        event.rulesConcept === 'anm_forced_elimination' &&
        event.payload &&
        event.payload.flowId === 'fe_loop_intro' &&
        event.payload.stepIndex === 4 &&
        event.payload.topic === 'forced_elimination'
    );

    expect(feLoopStep4Events.length).toBeGreaterThanOrEqual(1);
  });

  it('surfaces a structural stalemate teaching step for the victory_stalemate topic', async () => {
    render(
      <TeachingOverlay
        topic="victory_stalemate"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const stepButtons = await screen.findAllByTestId('teaching-related-step');
    const hasStructuralStalemateStep = stepButtons.some((button) => {
      const text = (button.textContent || '').replace(/\s+/g, ' ');
      return text.includes('structural_stalemate_intro');
    });

    expect(hasStructuralStalemateStep).toBe(true);
  });

  it('surfaces mini-region intro teaching steps for the territory topic', async () => {
    render(
      <TeachingOverlay topic="territory" isOpen={true} onClose={() => {}} position="center" />
    );

    const stepButtons = await screen.findAllByTestId('teaching-related-step');
    const hasMiniRegionStep = stepButtons.some((button) => {
      const text = (button.textContent || '').replace(/\s+/g, ' ');
      return text.includes('mini_region_intro');
    });

    expect(hasMiniRegionStep).toBe(true);
  });
});
