import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TeachingOverlay } from '../../src/client/components/TeachingOverlay';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';

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
    expect(event.payload).toEqual(
      expect.objectContaining({
        startedAutomatically: true,
        topic: 'active_no_moves',
      })
    );
  });
});
