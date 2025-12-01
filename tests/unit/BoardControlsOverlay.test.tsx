import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardControlsOverlay } from '../../src/client/components/BoardControlsOverlay';

describe('BoardControlsOverlay', () => {
  it('renders basic controls and keyboard shortcuts in backend mode', () => {
    const onClose = jest.fn();

    render(<BoardControlsOverlay mode="backend" onClose={onClose} />);

    expect(screen.getByTestId('board-controls-overlay')).toBeInTheDocument();
    expect(
      screen.getByTestId('board-controls-basic-section')
    ).toBeInTheDocument();
    expect(
      screen.getByTestId('board-controls-keyboard-section')
    ).toBeInTheDocument();
    expect(screen.getByText(/\?/i)).toBeInTheDocument();
  });

  it('renders sandbox touch controls and touch panel details when enabled', () => {
    const onClose = jest.fn();

    render(
      <BoardControlsOverlay
        mode="sandbox"
        hasTouchControlsPanel
        onClose={onClose}
      />
    );

    expect(
      screen.getByTestId('board-controls-sandbox-section')
    ).toBeInTheDocument();

    const panelHeadings = screen.getAllByText(/Sandbox touch controls panel/i);
    expect(panelHeadings.length).toBeGreaterThan(0);

    expect(screen.getByText(/Clear selection/i)).toBeInTheDocument();
    expect(screen.getByText(/Finish move/i)).toBeInTheDocument();
    expect(screen.getByText(/Show valid targets/i)).toBeInTheDocument();
    expect(screen.getByText(/Show movement grid/i)).toBeInTheDocument();
  });

  it('invokes onClose when the close button is clicked', () => {
    const onClose = jest.fn();

    render(<BoardControlsOverlay mode="backend" onClose={onClose} />);

    fireEvent.click(screen.getByTestId('board-controls-close-button'));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});