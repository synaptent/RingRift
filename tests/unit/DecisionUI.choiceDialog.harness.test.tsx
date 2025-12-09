import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ChoiceDialog } from '../../src/client/components/ChoiceDialog';
import type {
  CaptureDirectionChoice,
  LineRewardChoice,
  RegionOrderChoice,
  RingEliminationChoice,
  LineOrderChoice,
} from '../../src/shared/types/game';
import { usePendingChoice, useGameActions } from '../../src/client/hooks/useGameActions';

jest.mock('../../src/client/hooks/useGameActions');

type MockedUsePendingChoice = jest.MockedFunction<typeof usePendingChoice>;
type MockedUseGameActions = jest.MockedFunction<typeof useGameActions>;

const mockedUsePendingChoice = usePendingChoice as MockedUsePendingChoice;
const mockedUseGameActions = useGameActions as MockedUseGameActions;

function DecisionUIHarness() {
  const {
    pendingChoice,
    pendingChoiceView,
    choiceDeadline,
    reconciledDecisionTimeRemainingMs,
    decisionIsServerCapped,
  } = usePendingChoice();
  const { respondToChoice } = useGameActions();

  return (
    <ChoiceDialog
      choice={pendingChoice}
      choiceViewModel={pendingChoiceView?.viewModel}
      deadline={choiceDeadline}
      timeRemainingMs={reconciledDecisionTimeRemainingMs}
      isServerCapped={decisionIsServerCapped}
      onSelectOption={(choice, option) => respondToChoice(choice, option)}
    />
  );
}

describe('DecisionUI harness → ChoiceDialog integration', () => {
  const respondToChoice = jest.fn();

  const captureChoice: CaptureDirectionChoice = {
    id: 'choice-capture',
    gameId: 'game-123',
    playerNumber: 1,
    type: 'capture_direction',
    prompt: 'Choose capture direction',
    timeoutMs: 30_000,
    options: [
      {
        targetPosition: { x: 2, y: 2 },
        landingPosition: { x: 3, y: 3 },
        capturedCapHeight: 2,
      },
      {
        targetPosition: { x: 2, y: 4 },
        landingPosition: { x: 2, y: 5 },
        capturedCapHeight: 1,
      },
    ],
  };

  const lineRewardChoice: LineRewardChoice = {
    id: 'choice-line-reward',
    gameId: 'game-123',
    playerNumber: 1,
    type: 'line_reward_option',
    prompt: 'Choose line reward',
    timeoutMs: 20_000,
    options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
  };

  const regionOrderChoice: RegionOrderChoice = {
    id: 'choice-region',
    gameId: 'game-123',
    playerNumber: 1,
    type: 'region_order',
    prompt: 'Choose region order',
    timeoutMs: 25_000,
    options: [
      {
        regionId: 'region-a',
        size: 4,
        representativePosition: { x: 0, y: 0 },
        moveId: 'region-a-move',
      },
      {
        regionId: 'region-b',
        size: 2,
        representativePosition: { x: 1, y: 1 },
        moveId: 'region-b-move',
      },
      {
        regionId: 'skip',
        size: 0,
        representativePosition: { x: 0, y: 0 },
        moveId: 'skip-move',
      },
    ],
  };

  const ringEliminationChoice: RingEliminationChoice = {
    id: 'choice-ring-elim',
    gameId: 'game-123',
    playerNumber: 1,
    type: 'ring_elimination',
    prompt: 'Choose elimination target',
    timeoutMs: 15_000,
    options: [
      { stackPosition: { x: 1, y: 1 }, capHeight: 2, totalHeight: 3, moveId: 'elim-1' },
      { stackPosition: { x: 2, y: 2 }, capHeight: 1, totalHeight: 2, moveId: 'elim-2' },
    ],
  };

  const lineOrderChoice: LineOrderChoice = {
    id: 'choice-line-order',
    gameId: 'game-123',
    playerNumber: 1,
    type: 'line_order',
    prompt: 'Choose which line to process first',
    timeoutMs: 12_000,
    options: [
      {
        lineId: 'line-a',
        markerPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
          { x: 2, y: 2 },
        ],
        moveId: 'line-a-move',
      },
      {
        lineId: 'line-b',
        markerPositions: [
          { x: 0, y: 1 },
          { x: 1, y: 2 },
          { x: 2, y: 3 },
        ],
        moveId: 'line-b-move',
      },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: captureChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: null,
      reconciledDecisionTimeRemainingMs: 25_000,
    } as any);
    mockedUseGameActions.mockReturnValue({
      respondToChoice,
      respondToDecisionTimeout: jest.fn(),
      sendChatMessage: jest.fn(),
    } as any);
  });

  it('renders pending capture_direction choice and routes selection through respondToChoice', async () => {
    render(<DecisionUIHarness />);

    const options = await screen.findAllByText(/Choose capture direction/i);
    expect(options.length).toBeGreaterThan(0);
    const option = screen.getByText(/Direction 1/);
    fireEvent.click(option);

    expect(respondToChoice).toHaveBeenCalledTimes(1);
    const [choiceArg, optionArg] = respondToChoice.mock.calls[0];
    expect(choiceArg).toEqual(captureChoice);
    expect(optionArg).toEqual(captureChoice.options[0]);
  });

  it('renders line_reward_option choice and routes selection', async () => {
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineRewardChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: null,
      reconciledDecisionTimeRemainingMs: 18_000,
    } as any);

    render(<DecisionUIHarness />);

    expect(await screen.findAllByText(/Choose line reward/i)).toHaveLength(2);
    fireEvent.click(screen.getByText(/Full Collapse/i));

    expect(respondToChoice).toHaveBeenCalledTimes(1);
    const [choiceArg, optionArg] = respondToChoice.mock.calls[0];
    expect(choiceArg).toEqual(lineRewardChoice);
    expect(optionArg).toBe('option_1_collapse_all_and_eliminate');
  });

  it('renders region_order choice with skip option and routes selection', async () => {
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: regionOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: null,
      reconciledDecisionTimeRemainingMs: 15_000,
    } as any);

    render(<DecisionUIHarness />);

    expect(await screen.findByText(/Choose region order/i)).toBeInTheDocument();
    const skipButton = screen.getByText(/Skip territory processing for this turn/i);
    fireEvent.click(skipButton);

    expect(respondToChoice).toHaveBeenCalledTimes(1);
    const [choiceArg, optionArg] = respondToChoice.mock.calls[0];
    expect(choiceArg).toEqual(regionOrderChoice);
    expect(optionArg).toEqual(regionOrderChoice.options[2]);
  });

  it('renders ring_elimination choice and routes stack selection', async () => {
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: ringEliminationChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: null,
      reconciledDecisionTimeRemainingMs: 12_000,
    } as any);

    render(<DecisionUIHarness />);

    expect(await screen.findByText(/Choose elimination target/i)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/Stack at \(1, 1\)/i));

    expect(respondToChoice).toHaveBeenCalledTimes(1);
    const [choiceArg, optionArg] = respondToChoice.mock.calls[0];
    expect(choiceArg).toEqual(ringEliminationChoice);
    expect(optionArg).toEqual(ringEliminationChoice.options[0]);
  });

  it('renders line_order choice and routes line selection', async () => {
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: null,
      reconciledDecisionTimeRemainingMs: 10_000,
    } as any);

    render(<DecisionUIHarness />);

    expect(await screen.findByText(/Choose which line to process first/i)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/Line 2/i));

    expect(respondToChoice).toHaveBeenCalledTimes(1);
    const [choiceArg, optionArg] = respondToChoice.mock.calls[0];
    expect(choiceArg).toEqual(lineOrderChoice);
    expect(optionArg).toEqual(lineOrderChoice.options[1]);
  });

  it('shows countdown when deadline/timeRemainingMs are provided', async () => {
    const deadline = Date.now() + 5000;
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: deadline,
      reconciledDecisionTimeRemainingMs: 5000,
    } as any);

    render(<DecisionUIHarness />);

    expect(await screen.findByText(/Respond within/i)).toBeInTheDocument();
    expect(screen.getByText('5s')).toBeInTheDocument();
  });

  it('applies critical severity when decision time remaining is low', async () => {
    const deadline = Date.now() + 2500;
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: deadline,
      reconciledDecisionTimeRemainingMs: 2500,
    } as any);

    render(<DecisionUIHarness />);

    const countdown = await screen.findByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-severity', 'critical');
    expect(screen.getByText('3s')).toBeInTheDocument();
  });

  it('applies warning severity when decision time remaining is between 3s and 10s', async () => {
    const deadline = Date.now() + 8000;
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: deadline,
      reconciledDecisionTimeRemainingMs: 8000,
    } as any);

    render(<DecisionUIHarness />);

    const countdown = await screen.findByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-severity', 'warning');
    expect(screen.getByText('8s')).toBeInTheDocument();
  });

  it('marks countdown as server-capped and updates copy when server reduces deadline', async () => {
    const deadline = Date.now() + 3_000;
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: deadline,
      reconciledDecisionTimeRemainingMs: 2_500,
      decisionIsServerCapped: true,
    } as any);

    render(<DecisionUIHarness />);

    const countdown = await screen.findByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-server-capped', 'true');
    expect(screen.getByText(/Server deadline – respond within/)).toBeInTheDocument();
  });

  it('surfaces countdown severity for short timers', async () => {
    const deadline = Date.now() + 2500;
    mockedUsePendingChoice.mockReturnValue({
      pendingChoice: lineOrderChoice,
      pendingChoiceView: { viewModel: undefined },
      choiceDeadline: deadline,
      reconciledDecisionTimeRemainingMs: 2500,
    } as any);

    render(<DecisionUIHarness />);

    const countdown = await screen.findByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-severity', 'critical');
    expect(screen.getByText(/Respond within/i)).toBeInTheDocument();
  });
});
