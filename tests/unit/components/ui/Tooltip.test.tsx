import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Tooltip } from '../../../../src/client/components/ui/Tooltip';

describe('Tooltip', () => {
  it('renders trigger element without tooltip by default', () => {
    render(
      <Tooltip content="Turn counts full rounds; Move counts individual actions.">
        <button type="button">Trigger</button>
      </Tooltip>
    );

    const trigger = screen.getByRole('button', { name: 'Trigger' });
    expect(trigger).toBeInTheDocument();
    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  it('shows tooltip on hover and hides on mouse leave', () => {
    render(
      <Tooltip content="Hover help">
        <button type="button">Hover me</button>
      </Tooltip>
    );

    const trigger = screen.getByRole('button', { name: 'Hover me' });

    fireEvent.mouseEnter(trigger);
    const tooltip = screen.getByRole('tooltip');
    expect(tooltip).toBeInTheDocument();
    expect(tooltip).toHaveTextContent('Hover help');

    fireEvent.mouseLeave(trigger);
    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  it('shows tooltip on focus and hides on blur for keyboard users', () => {
    render(
      <Tooltip content="Keyboard help">
        <button type="button">Focus me</button>
      </Tooltip>
    );

    const trigger = screen.getByRole('button', { name: 'Focus me' });

    fireEvent.focus(trigger);
    const tooltip = screen.getByRole('tooltip');
    expect(tooltip).toBeInTheDocument();
    expect(tooltip).toHaveTextContent('Keyboard help');

    fireEvent.blur(trigger);
    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  it('preserves existing event handlers on the trigger element', () => {
    const mouseEnter = jest.fn();
    const mouseLeave = jest.fn();
    const focus = jest.fn();
    const blur = jest.fn();

    render(
      <Tooltip content="Handlers test">
        <button
          type="button"
          onMouseEnter={mouseEnter}
          onMouseLeave={mouseLeave}
          onFocus={focus}
          onBlur={blur}
        >
          With handlers
        </button>
      </Tooltip>
    );

    const trigger = screen.getByRole('button', { name: 'With handlers' });

    fireEvent.mouseEnter(trigger);
    fireEvent.mouseLeave(trigger);
    fireEvent.focus(trigger);
    fireEvent.blur(trigger);

    expect(mouseEnter).toHaveBeenCalledTimes(1);
    expect(mouseLeave).toHaveBeenCalledTimes(1);
    expect(focus).toHaveBeenCalledTimes(1);
    expect(blur).toHaveBeenCalledTimes(1);
  });
});
