import React from 'react';
import '@testing-library/jest-dom';
import { render, screen, fireEvent } from '@testing-library/react';
import { TeachingOverlay } from '../../src/client/components/TeachingOverlay';
import type {
  RulesUxContext,
  RulesUxWeirdStateType,
} from '../../src/shared/telemetry/rulesUxEvents';
import type { RulesWeirdStateReasonCode } from '../../src/shared/engine/weirdStateReasons';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';

jest.mock('../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../src/client/utils/rulesUxTelemetry');
  return {
    __esModule: true,
    ...actual,
    // Replace logRulesUxEvent with a Jest mock so we can assert on the
    // final telemetry envelope without performing real network calls.
    logRulesUxEvent: jest.fn(),
    // Stabilise teachingFlowId generation for deterministic assertions.
    newTeachingFlowId: jest.fn(() => 'teaching-flow-1'),
  };
});

const mockLogRulesUxEvent = rulesUxTelemetry.logRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logRulesUxEvent
>;

function createWeirdStateOverlayContext(
  overrides: Partial<{
    reasonCode: RulesWeirdStateReasonCode;
    rulesContext: RulesUxContext;
    weirdStateType: RulesUxWeirdStateType;
    boardType: 'square8' | 'square19' | 'hexagonal';
    numPlayers: number;
    isRanked: boolean;
    isSandbox: boolean;
    overlaySessionId: string;
  }> = {}
) {
  return {
    reasonCode: 'ANM_MOVEMENT_FE_BLOCKED' as RulesWeirdStateReasonCode,
    rulesContext: 'anm_forced_elimination' as RulesUxContext,
    weirdStateType: 'active-no-moves-movement' as RulesUxWeirdStateType,
    boardType: 'square8' as const,
    numPlayers: 2,
    isRanked: false,
    isSandbox: true,
    overlaySessionId: 'overlay-session-1',
    ...overrides,
  };
}

describe('TeachingOverlay – rules‑UX telemetry', () => {
  beforeEach(() => {
    mockLogRulesUxEvent.mockReset();
  });

  it('emits weird_state_overlay_shown and weird_state_overlay_dismiss for weird-state context', () => {
    const ctx = createWeirdStateOverlayContext();

    const { rerender } = render(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={false}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    // Open the overlay – should emit weird_state_overlay_shown once.
    rerender(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    expect(mockLogRulesUxEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'weird_state_overlay_shown',
        source: 'teaching_overlay',
        boardType: 'square8',
        numPlayers: 2,
        rulesContext: 'anm_forced_elimination',
        weirdStateType: 'active-no-moves-movement',
        reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
        overlaySessionId: 'overlay-session-1',
      })
    );

    mockLogRulesUxEvent.mockClear();

    // Close the overlay – should emit weird_state_overlay_dismiss once.
    rerender(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={false}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    expect(mockLogRulesUxEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'weird_state_overlay_dismiss',
        source: 'teaching_overlay',
        boardType: 'square8',
        numPlayers: 2,
        rulesContext: 'anm_forced_elimination',
        weirdStateType: 'active-no-moves-movement',
        reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
        overlaySessionId: 'overlay-session-1',
      })
    );
  });

  it('auto-selects a teaching step for weird-state context and emits teaching_step_started', () => {
    const ctx = createWeirdStateOverlayContext();

    render(
      <TeachingOverlay
        topic="active_no_moves"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    // teaching_step_started should be emitted for the first related scenario.
    expect(
      mockLogRulesUxEvent.mock.calls.some(
        ([event]) =>
          (event as any).type === 'teaching_step_started' &&
          (event as any).source === 'teaching_overlay' &&
          (event as any).rulesContext === 'anm_forced_elimination' &&
          (event as any).scenarioId === 'teaching.fe_loop.step_1'
      )
    ).toBe(true);
  });

  it('emits teaching_step_completed when marking a step as understood', () => {
    render(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    // Select the first related teaching step.
    const stepButtons = screen.getAllByTestId('teaching-related-step');
    expect(stepButtons.length).toBeGreaterThan(0);
    fireEvent.click(stepButtons[0]);

    mockLogRulesUxEvent.mockClear();

    // Click "Mark as understood" in the step details panel.
    const markUnderstood = screen.getByRole('button', { name: /mark as understood/i });
    fireEvent.click(markUnderstood);

    expect(
      mockLogRulesUxEvent.mock.calls.some(
        ([event]) =>
          (event as any).type === 'teaching_step_completed' &&
          (event as any).source === 'teaching_overlay' &&
          (event as any).payload?.completionAction === 'mark_understood'
      )
    ).toBe(true);
  });
});
