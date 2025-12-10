import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TeachingOverlay } from '../../../src/client/components/TeachingOverlay';
import { logRulesUxEvent } from '../../../src/client/utils/rulesUxTelemetry';

jest.mock('../../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../../src/client/utils/rulesUxTelemetry');
  return {
    ...actual,
    logRulesUxEvent: jest.fn().mockResolvedValue(undefined),
    newTeachingFlowId: jest.fn(() => 'test-flow-id'),
  };
});

describe('TeachingOverlay basic rendering', () => {
  it('renders when open and calls onClose', () => {
    const onClose = jest.fn();

    render(
      <TeachingOverlay topic="territory" isOpen={true} onClose={onClose} position="center" />
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /territory/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes on Escape key', () => {
    const onClose = jest.fn();

    render(<TeachingOverlay topic="territory" isOpen={true} onClose={onClose} position="center" />);

    fireEvent.keyDown(window, { key: 'Escape', code: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('emits weird-state telemetry lifecycle and auto-selects related scenario', async () => {
    const onClose = jest.fn();
    const telemetryMock = logRulesUxEvent as jest.Mock;

    const weirdStateOverlayContext = {
      reasonCode: 'ANM_TERRITORY_NO_ACTIONS',
      rulesContext: 'territory_mini_region',
      boardType: 'square8' as const,
      numPlayers: 2 as const,
      overlaySessionId: 'overlay-session-1',
    };

    const { rerender } = render(
      <TeachingOverlay
        topic="territory"
        isOpen={true}
        onClose={onClose}
        position="center"
        weirdStateOverlayContext={weirdStateOverlayContext}
      />
    );

    await waitFor(() => {
      expect(telemetryMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'weird_state_overlay_shown',
          overlaySessionId: 'overlay-session-1',
          reasonCode: 'ANM_TERRITORY_NO_ACTIONS',
          rulesContext: 'territory_mini_region',
        })
      );
    });

    await waitFor(() => {
      expect(telemetryMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'teaching_step_started',
          overlaySessionId: 'overlay-session-1',
          teachingFlowId: 'test-flow-id',
          rulesConcept: 'territory_mini_region',
          payload: expect.objectContaining({
            flowId: 'mini_region_intro',
            stepIndex: 1,
            topic: 'territory',
            startedAutomatically: true,
          }),
        })
      );
    });

    telemetryMock.mockClear();

    rerender(
      <TeachingOverlay
        topic="territory"
        isOpen={false}
        onClose={onClose}
        position="center"
        weirdStateOverlayContext={weirdStateOverlayContext}
      />
    );

    await waitFor(() => {
      expect(telemetryMock).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'weird_state_overlay_dismiss',
          overlaySessionId: 'overlay-session-1',
          rulesContext: 'territory_mini_region',
        })
      );
    });
  });
});
