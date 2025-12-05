import React from 'react';
import { render, waitFor } from '@testing-library/react';
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

describe('TeachingOverlay – weird-state overlay lifecycle telemetry', () => {
  beforeEach(() => {
    mockLogRulesUxEvent.mockReset();
  });

  it('emits weird_state_overlay_shown on mount when opened with a weird-state context', async () => {
    const ctx = {
      reasonCode: 'ANM_MOVEMENT_FE_BLOCKED' as const,
      rulesContext: 'anm_forced_elimination' as const,
      weirdStateType: 'active-no-moves-movement' as const,
      boardType: 'square8' as const,
      numPlayers: 2,
      isRanked: true,
      isSandbox: false,
      overlaySessionId: 'overlay-session-test-1',
    };

    render(
      <TeachingOverlay
        topic="active_no_moves"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    await waitFor(() => {
      expect(mockLogRulesUxEvent).toHaveBeenCalled();
    });

    const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
    const shown = events.find((e) => e.type === 'weird_state_overlay_shown');
    expect(shown).toBeDefined();

    expect(shown).toMatchObject({
      type: 'weird_state_overlay_shown',
      source: 'teaching_overlay',
      boardType: 'square8',
      numPlayers: 2,
      rulesContext: 'anm_forced_elimination',
      weirdStateType: 'active-no-moves-movement',
      reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
      isRanked: true,
      isSandbox: false,
      overlaySessionId: 'overlay-session-test-1',
    });
  });

  it('emits weird_state_overlay_dismiss with the same overlaySessionId when closed', async () => {
    const ctx = {
      reasonCode: 'ANM_MOVEMENT_FE_BLOCKED' as const,
      rulesContext: 'anm_forced_elimination' as const,
      weirdStateType: 'active-no-moves-movement' as const,
      boardType: 'square8' as const,
      numPlayers: 2,
      isRanked: false,
      isSandbox: true,
      overlaySessionId: 'overlay-session-test-2',
    };

    const { rerender } = render(
      <TeachingOverlay
        topic="active_no_moves"
        isOpen={true}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
      expect(events.some((e) => e.type === 'weird_state_overlay_shown')).toBe(true);
    });

    mockLogRulesUxEvent.mockClear();

    rerender(
      <TeachingOverlay
        topic="active_no_moves"
        isOpen={false}
        onClose={() => {}}
        position="center"
        weirdStateOverlayContext={ctx}
      />
    );

    await waitFor(() => {
      expect(mockLogRulesUxEvent).toHaveBeenCalled();
    });

    const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
    const dismiss = events.find((e) => e.type === 'weird_state_overlay_dismiss');
    expect(dismiss).toBeDefined();

    expect(dismiss).toMatchObject({
      type: 'weird_state_overlay_dismiss',
      source: 'teaching_overlay',
      boardType: 'square8',
      numPlayers: 2,
      rulesContext: 'anm_forced_elimination',
      weirdStateType: 'active-no-moves-movement',
      reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
      isRanked: false,
      isSandbox: true,
      overlaySessionId: 'overlay-session-test-2',
    });
  });
});
