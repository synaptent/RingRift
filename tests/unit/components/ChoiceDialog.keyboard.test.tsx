import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ChoiceDialog } from '../../../src/client/components/ChoiceDialog';
import type { LineOrderChoice } from '../../../src/shared/types/game';

const lineOrderChoice: LineOrderChoice = {
  id: 'choice-line-order',
  gameId: 'g1',
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

describe('ChoiceDialog keyboard + cancel interactions', () => {
  it('cycles option focus with arrow keys and respects Escape to cancel', async () => {
    const onSelectOption = jest.fn();
    const onCancel = jest.fn();

    render(
      <ChoiceDialog
        choice={lineOrderChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 12_000}
        timeRemainingMs={12_000}
        isServerCapped={false}
        onSelectOption={onSelectOption}
        onCancel={onCancel}
      />
    );

    const dialog = screen.getByRole('dialog');
    const buttons = screen.getAllByRole('option');

    // Initial focus should be on the first option.
    expect(document.activeElement).toBe(buttons[0]);
    expect(buttons[0]).toHaveAttribute('aria-selected', 'true');

    // Spy on focus to verify navigation (JSDOM doesn't reliably update activeElement)
    const focusSpy = jest.spyOn(buttons[1], 'focus');

    // ArrowDown should move focus/selection to the next option.
    await act(async () => {
      fireEvent.keyDown(dialog, { key: 'ArrowDown' });
    });

    // Verify focus() was called on the next option
    expect(focusSpy).toHaveBeenCalled();
    focusSpy.mockRestore();

    // Escape should trigger onCancel.
    await act(async () => {
      fireEvent.keyDown(dialog, { key: 'Escape' });
    });
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onSelectOption).not.toHaveBeenCalled();
  });

  it('wraps focus to last option when pressing ArrowUp at the top', async () => {
    render(
      <ChoiceDialog
        choice={lineOrderChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 12_000}
        timeRemainingMs={12_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
        onCancel={jest.fn()}
      />
    );

    const dialog = screen.getByRole('dialog');
    const buttons = screen.getAllByRole('option');

    const lastIndex = buttons.length - 1;
    const lastFocusSpy = jest.spyOn(buttons[lastIndex], 'focus');

    await act(async () => {
      fireEvent.keyDown(dialog, { key: 'ArrowUp' });
    });

    expect(lastFocusSpy).toHaveBeenCalled();
    lastFocusSpy.mockRestore();
  });

  it('invokes onCancel when cancel button is clicked and does not submit options', () => {
    const onSelectOption = jest.fn();
    const onCancel = jest.fn();

    render(
      <ChoiceDialog
        choice={lineOrderChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 5_000}
        timeRemainingMs={5_000}
        isServerCapped={false}
        onSelectOption={onSelectOption}
        onCancel={onCancel}
      />
    );

    fireEvent.click(screen.getByText(/Cancel/i));
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onSelectOption).not.toHaveBeenCalled();
  });

  it('traps focus with Tab from last focusable back to first', async () => {
    render(
      <ChoiceDialog
        choice={lineOrderChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 12_000}
        timeRemainingMs={12_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
        onCancel={jest.fn()}
      />
    );

    const dialog = screen.getByRole('dialog');
    const optionButtons = screen.getAllByRole('option');
    const cancelButton = screen.getByText(/Cancel/i);

    // Move focus to the last focusable element (cancel button).
    cancelButton.focus();
    expect(document.activeElement).toBe(cancelButton);

    const firstFocusSpy = jest.spyOn(optionButtons[0], 'focus');

    await act(async () => {
      fireEvent.keyDown(dialog, { key: 'Tab' });
    });

    expect(firstFocusSpy).toHaveBeenCalled();
    firstFocusSpy.mockRestore();
  });

  it('traps focus with Shift+Tab from first focusable back to last', async () => {
    render(
      <ChoiceDialog
        choice={lineOrderChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 12_000}
        timeRemainingMs={12_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
        onCancel={jest.fn()}
      />
    );

    const dialog = screen.getByRole('dialog');
    const optionButtons = screen.getAllByRole('option');
    const cancelButton = screen.getByText(/Cancel/i);

    optionButtons[0].focus();
    expect(document.activeElement).toBe(optionButtons[0]);

    const cancelFocusSpy = jest.spyOn(cancelButton, 'focus');

    await act(async () => {
      fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
    });

    expect(cancelFocusSpy).toHaveBeenCalled();
    cancelFocusSpy.mockRestore();
  });

  it('resets focused option when the choice changes', async () => {
    const { rerender } = render(
      <ChoiceDialog
        choice={lineOrderChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 12_000}
        timeRemainingMs={12_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
        onCancel={jest.fn()}
      />
    );

    const dialog = screen.getByRole('dialog');
    const options = screen.getAllByRole('option');

    // Move focus to the second option via focus event (invokes onFocus handler).
    await act(async () => {
      fireEvent.focus(options[1]);
    });
    expect(options[1]).toHaveAttribute('aria-selected', 'true');

    // Rerender with a new choice id should reset selection to first option.
    const newChoice = { ...lineOrderChoice, id: 'choice-line-order-2' };
    rerender(
      <ChoiceDialog
        choice={newChoice}
        choiceViewModel={undefined}
        deadline={Date.now() + 12_000}
        timeRemainingMs={12_000}
        isServerCapped={false}
        onSelectOption={jest.fn()}
        onCancel={jest.fn()}
      />
    );

    const newOptions = screen.getAllByRole('option');
    expect(newOptions[0]).toHaveAttribute('aria-selected', 'true');
  });
});
