import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { OnboardingModal } from '../../../src/client/components/OnboardingModal';
import * as rulesUxTelemetry from '../../../src/client/utils/rulesUxTelemetry';

jest.mock('../../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../../src/client/utils/rulesUxTelemetry');
  return {
    __esModule: true,
    ...actual,
    // Mock telemetry helpers so we can assert on envelopes without network calls.
    sendRulesUxEvent: jest.fn(),
    logHelpOpenEvent: jest.fn(),
    // Stabilise helpSessionId so we can assert correlations deterministically.
    newHelpSessionId: jest.fn(() => 'help-session-onboarding-1'),
  };
});

const mockSendRulesUxEvent = rulesUxTelemetry.sendRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.sendRulesUxEvent
>;
const mockLogHelpOpenEvent = rulesUxTelemetry.logHelpOpenEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logHelpOpenEvent
>;

describe('OnboardingModal – canonical copy & rules‑UX telemetry', () => {
  const defaultProps = {
    isOpen: true,
    onClose: () => {},
    onStartTutorial: () => {},
  };

  beforeEach(() => {
    mockSendRulesUxEvent.mockReset();
    mockLogHelpOpenEvent.mockReset();
  });

  it('renders canonical Ring Elimination victory copy from UX_RULES_COPY_SPEC', () => {
    render(<OnboardingModal {...defaultProps} />);

    // Advance to the Victory step (Welcome → Phases → Victory).
    const nextButton = screen.getByText('Next');
    fireEvent.click(nextButton);
    fireEvent.click(nextButton);

    // Victory step now uses short descriptions from VICTORY_SHORT_DESCRIPTIONS
    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    expect(text).toMatch(/Ring Elimination/i);
    expect(text).toMatch(/elimination threshold/i);
    expect(text).toMatch(/Territory Control/i);
    expect(text).toMatch(/Last Player Standing/i);
  });

  it('emits a spec-aligned help_open and legacy rules_help_open event when opened', async () => {
    render(<OnboardingModal {...defaultProps} />);

    await waitFor(() => {
      expect(mockLogHelpOpenEvent).toHaveBeenCalled();
      expect(mockSendRulesUxEvent).toHaveBeenCalled();
    });

    // New spec-aligned help_open via logHelpOpenEvent options
    const helpOpenCall = mockLogHelpOpenEvent.mock.calls[0]?.[0];
    expect(helpOpenCall).toMatchObject({
      boardType: 'square8',
      numPlayers: 2,
      difficulty: 'tutorial',
      source: 'sandbox',
      entrypoint: 'sandbox_toolbar_help',
      isSandbox: true,
      helpSessionId: 'help-session-onboarding-1',
    });

    // Legacy metrics event for backwards compatibility
    const rulesHelpOpen = mockSendRulesUxEvent.mock.calls.find(
      ([event]) => (event as any).type === 'rules_help_open'
    );
    expect(rulesHelpOpen).toBeDefined();
    expect(rulesHelpOpen?.[0]).toMatchObject({
      type: 'rules_help_open',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'onboarding',
      rulesConcept: 'board_intro_square8',
      isSandbox: true,
    });
  });

  it('emits a help_topic_view for the overall onboarding summary when closed', async () => {
    const { rerender } = render(
      <OnboardingModal isOpen={true} onClose={() => {}} onStartTutorial={() => {}} />
    );

    // Allow the open-side telemetry to fire, then clear for a focused assertion.
    await waitFor(() => {
      expect(mockSendRulesUxEvent).toHaveBeenCalled();
    });
    mockSendRulesUxEvent.mockClear();

    // Close the modal – this should trigger a help_topic_view with topic_id "onboarding.summary".
    rerender(<OnboardingModal isOpen={false} onClose={() => {}} onStartTutorial={() => {}} />);

    await waitFor(() => {
      expect(mockSendRulesUxEvent).toHaveBeenCalled();
    });

    const topicView = mockSendRulesUxEvent.mock.calls.find(
      ([event]) => (event as any).type === 'help_topic_view'
    );
    expect(topicView).toBeDefined();
    expect(topicView?.[0]).toMatchObject({
      type: 'help_topic_view',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'onboarding',
      rulesConcept: 'board_intro_square8',
      isSandbox: true,
      helpSessionId: 'help-session-onboarding-1',
    });

    const payload = (topicView?.[0] as any).payload;
    expect(payload).toMatchObject({
      topic_id: 'onboarding.summary',
    });
  });

  it('emits help_topic_view events when individual victory concepts are clicked', async () => {
    render(<OnboardingModal {...defaultProps} />);

    // Move to the Victory step so the victory cards are visible.
    const nextButton = screen.getByText('Next');
    fireEvent.click(nextButton);
    fireEvent.click(nextButton);

    // Wait for initial open telemetry, then clear to focus on click events.
    await waitFor(() => {
      expect(mockSendRulesUxEvent).toHaveBeenCalled();
    });
    mockSendRulesUxEvent.mockClear();

    const eliminationCard = screen.getByText('Ring Elimination');
    fireEvent.click(eliminationCard);

    await waitFor(() => {
      expect(mockSendRulesUxEvent).toHaveBeenCalled();
    });

    const calls = mockSendRulesUxEvent.mock.calls.map(([event]) => event as any);
    const eliminationTopicView = calls.find(
      (e) =>
        e.type === 'help_topic_view' && e.payload?.topic_id === 'onboarding.victory.elimination'
    );

    expect(eliminationTopicView).toBeDefined();
    expect(eliminationTopicView).toMatchObject({
      type: 'help_topic_view',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'onboarding',
      rulesConcept: 'board_intro_square8',
      isSandbox: true,
    });
  });
});
