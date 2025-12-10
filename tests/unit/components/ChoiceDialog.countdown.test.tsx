import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ChoiceDialog } from '../../../src/client/components/ChoiceDialog';
import type { LineRewardChoice } from '../../../src/shared/types/game';

function buildLineRewardChoice(): LineRewardChoice {
  return {
    id: 'choice-line-reward',
    gameId: 'g1',
    playerNumber: 1,
    type: 'line_reward_option',
    prompt: 'Choose line reward',
    timeoutMs: 20_000,
    options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
  };
}

describe('ChoiceDialog countdown + server cap styling', () => {
  it('renders countdown with warning severity and server-cap label', () => {
    const choice = buildLineRewardChoice();

    render(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={undefined}
        deadline={Date.now() + 4_000}
        timeRemainingMs={4_000}
        isServerCapped={true}
        onSelectOption={jest.fn()}
      />
    );

    const countdown = screen.getByTestId('choice-countdown');
    expect(countdown).toBeInTheDocument();
    expect(countdown).toHaveAttribute('data-severity', 'warning');
    expect(countdown).toHaveAttribute('data-server-capped', 'true');
    expect(screen.getByText(/Server deadline – respond within/i)).toBeInTheDocument();
    expect(screen.getByText(/4s/)).toBeInTheDocument();

    // Progress bar width should reflect 4s remaining out of 20s (≈20%).
    const bar = screen.getByTestId('choice-countdown-bar');
    expect(bar).toHaveStyle({ width: '20%' });
  });

  it('renders critical severity when countdown is near expiry', () => {
    const choice = buildLineRewardChoice();

    render(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={undefined}
        deadline={Date.now() + 2_000}
        timeRemainingMs={2_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
      />
    );

    const countdown = screen.getByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-severity', 'critical');
    expect(screen.getByText(/Respond within/i)).toBeInTheDocument();
    expect(screen.getByText(/2s/)).toBeInTheDocument();
  });

  it('falls back to timeout copy when time remaining is unknown', () => {
    const choice = buildLineRewardChoice();

    render(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={undefined}
        deadline={Date.now() + 5_000}
        timeRemainingMs={null}
        isServerCapped={false}
        onSelectOption={jest.fn()}
      />
    );

    expect(screen.getByTestId('choice-countdown')).toBeInTheDocument();
    expect(screen.getByText(/Choice timeout active/i)).toBeInTheDocument();
  });

  it('marks normal severity when plenty of time remains', () => {
    const choice = buildLineRewardChoice();

    render(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={undefined}
        deadline={Date.now() + 20_000}
        timeRemainingMs={15_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
      />
    );

    const countdown = screen.getByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-severity', 'normal');
    expect(countdown).toHaveTextContent('15s');
  });

  it('clamps progress bar between 0% and 100%', () => {
    const choice = buildLineRewardChoice();

    const { rerender } = render(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={undefined}
        deadline={Date.now() + 25_000}
        timeRemainingMs={25_000} // greater than total timeout
        isServerCapped={false}
        onSelectOption={jest.fn()}
      />
    );

    let bar = screen.getByTestId('choice-countdown-bar');
    expect(bar).toHaveStyle({ width: '100%' });

    rerender(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={undefined}
        deadline={Date.now() - 1_000}
        timeRemainingMs={-1_000} // negative time -> clamp to 0
        isServerCapped={false}
        onSelectOption={jest.fn()}
      />
    );

    bar = screen.getByTestId('choice-countdown-bar');
    expect(bar).toHaveStyle({ width: '0%' });
  });
});
