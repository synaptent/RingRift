import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import {
  BoardControlsOverlay,
  BoardControlsOverlayMode,
} from '../../../src/client/components/BoardControlsOverlay';

describe('BoardControlsOverlay', () => {
  const defaultProps = {
    mode: 'backend' as BoardControlsOverlayMode,
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders as a modal dialog with proper ARIA attributes', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
      expect(dialog).toHaveAttribute('aria-labelledby', 'board-controls-title');
      expect(dialog).toHaveAttribute('aria-describedby', 'board-controls-description');
    });

    it('renders title and description', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('Board controls & shortcuts')).toBeInTheDocument();
      expect(
        screen.getByText(/How to select stacks, apply moves, and use keyboard shortcuts/i)
      ).toBeInTheDocument();
    });

    it('renders basic controls section', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByTestId('board-controls-basic-section')).toBeInTheDocument();
      expect(screen.getByText(/Basic mouse \/ touch controls/i)).toBeInTheDocument();
    });

    it('renders keyboard shortcuts section', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByTestId('board-controls-keyboard-section')).toBeInTheDocument();
      expect(screen.getByText(/Keyboard shortcuts \(desktop\)/i)).toBeInTheDocument();
    });

    it('renders close button', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByTestId('board-controls-close-button')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Close board controls/i })).toBeInTheDocument();
    });
  });

  describe('mode-specific rendering', () => {
    describe('backend mode', () => {
      it('renders Backend game badge', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="backend" />);

        expect(screen.getByText('Backend game')).toBeInTheDocument();
      });

      it('shows resign shortcut in keyboard section', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="backend" />);

        expect(screen.getByText(/resign \(backend games only\)/i)).toBeInTheDocument();
      });

      it('does not render sandbox touch controls section', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="backend" />);

        expect(screen.queryByTestId('board-controls-sandbox-section')).not.toBeInTheDocument();
      });

      it('shows ring placement instructions for non-spectators', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="backend" />);

        expect(screen.getByText(/ring placement/i)).toBeInTheDocument();
        expect(
          screen.getByText(/highlighted cells show legal ring locations/i)
        ).toBeInTheDocument();
      });
    });

    describe('sandbox mode', () => {
      it('renders Sandbox badge', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="sandbox" />);

        expect(screen.getByText('Sandbox')).toBeInTheDocument();
      });

      it('renders sandbox-specific title and description', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="sandbox" />);

        expect(screen.getByText('Sandbox board controls & shortcuts')).toBeInTheDocument();
        expect(
          screen.getByText(/How to tap, drag, and use the sandbox touch controls panel/i)
        ).toBeInTheDocument();
      });

      it('renders sandbox touch controls section', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="sandbox" />);

        expect(screen.getByTestId('board-controls-sandbox-section')).toBeInTheDocument();
        // Use heading role to be specific about the section heading
        expect(
          screen.getByRole('heading', { name: /Sandbox touch controls/i })
        ).toBeInTheDocument();
      });

      it('does not show resign shortcut', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="sandbox" />);

        expect(screen.queryByText(/resign \(backend games only\)/i)).not.toBeInTheDocument();
      });

      it('shows touch controls panel info when hasTouchControlsPanel is true', () => {
        render(
          <BoardControlsOverlay {...defaultProps} mode="sandbox" hasTouchControlsPanel={true} />
        );

        expect(screen.getByText('Sandbox touch controls panel')).toBeInTheDocument();
        expect(screen.getByText(/Clear selection/i)).toBeInTheDocument();
        expect(screen.getByText(/Finish move/i)).toBeInTheDocument();
        expect(screen.getByText(/Show valid targets/i)).toBeInTheDocument();
        expect(screen.getByText(/Show movement grid/i)).toBeInTheDocument();
      });

      it('does not show touch controls panel info when hasTouchControlsPanel is false', () => {
        render(
          <BoardControlsOverlay {...defaultProps} mode="sandbox" hasTouchControlsPanel={false} />
        );

        expect(screen.queryByText('Sandbox touch controls panel')).not.toBeInTheDocument();
      });
    });

    describe('spectator mode', () => {
      it('renders Spectator badge', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="spectator" />);

        expect(screen.getByText('Spectator')).toBeInTheDocument();
      });

      it('shows spectator-specific message in basic controls', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="spectator" />);

        expect(screen.getByText(/As a spectator the board is read-only/i)).toBeInTheDocument();
      });

      it('does not show ring placement instructions', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="spectator" />);

        expect(
          screen.queryByText(/highlighted cells show legal ring locations/i)
        ).not.toBeInTheDocument();
      });

      it('does not show resign shortcut', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="spectator" />);

        expect(screen.queryByText(/resign \(backend games only\)/i)).not.toBeInTheDocument();
      });

      it('does not show keyboard navigation accessibility message', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="spectator" />);

        expect(
          screen.queryByText(
            /Keyboard navigation allows you to play RingRift entirely without a mouse/i
          )
        ).not.toBeInTheDocument();
      });

      it('does not render sandbox touch controls section', () => {
        render(<BoardControlsOverlay {...defaultProps} mode="spectator" />);

        expect(screen.queryByTestId('board-controls-sandbox-section')).not.toBeInTheDocument();
      });
    });
  });

  describe('close interactions', () => {
    it('calls onClose when close button is clicked', async () => {
      const onClose = jest.fn();
      const user = userEvent.setup();
      render(<BoardControlsOverlay {...defaultProps} onClose={onClose} />);

      await user.click(screen.getByTestId('board-controls-close-button'));

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when Escape key is pressed', () => {
      const onClose = jest.fn();
      render(<BoardControlsOverlay {...defaultProps} onClose={onClose} />);

      fireEvent.keyDown(document, { key: 'Escape' });

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when backdrop is clicked', async () => {
      const onClose = jest.fn();
      const user = userEvent.setup();
      render(<BoardControlsOverlay {...defaultProps} onClose={onClose} />);

      // Click on the backdrop (the outer dialog element)
      await user.click(screen.getByTestId('board-controls-overlay'));

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('does not call onClose when clicking inside the card', async () => {
      const onClose = jest.fn();
      const user = userEvent.setup();
      render(<BoardControlsOverlay {...defaultProps} onClose={onClose} />);

      // Click on content inside the card
      await user.click(screen.getByText('Board controls & shortcuts'));

      expect(onClose).not.toHaveBeenCalled();
    });
  });

  describe('focus management', () => {
    it('focuses close button on mount', async () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByTestId('board-controls-close-button')).toHaveFocus();
      });
    });

    it('returns focus to previously focused element on unmount', async () => {
      const button = document.createElement('button');
      button.textContent = 'Previous focus';
      document.body.appendChild(button);
      button.focus();

      const { unmount } = render(<BoardControlsOverlay {...defaultProps} />);

      // Close button should have focus now
      await waitFor(() => {
        expect(screen.getByTestId('board-controls-close-button')).toHaveFocus();
      });

      unmount();

      expect(button).toHaveFocus();

      document.body.removeChild(button);
    });

    it('traps focus with Tab key - wraps from last to first', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      const closeButton = screen.getByTestId('board-controls-close-button');
      closeButton.focus();

      // Tab should stay within the dialog
      fireEvent.keyDown(document, { key: 'Tab' });

      // Focus should wrap to first focusable element (close button is only focusable)
      expect(closeButton).toHaveFocus();
    });

    it('traps focus with Shift+Tab - wraps from first to last', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      const closeButton = screen.getByTestId('board-controls-close-button');
      closeButton.focus();

      // Shift+Tab should wrap to last focusable (same element in this case)
      fireEvent.keyDown(document, { key: 'Tab', shiftKey: true });

      expect(closeButton).toHaveFocus();
    });
  });

  describe('keyboard shortcuts content', () => {
    it('lists arrow key navigation', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('Arrow keys')).toBeInTheDocument();
      expect(screen.getByText(/move focus around the board/i)).toBeInTheDocument();
    });

    it('lists Enter/Space activation', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      // Multiple 'Enter' and 'Space' text elements exist, use getAllByText
      expect(screen.getAllByText('Enter').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Space').length).toBeGreaterThan(0);
      expect(screen.getByText(/activate the focused cell/i)).toBeInTheDocument();
    });

    it('lists Esc for clearing selection', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('Esc')).toBeInTheDocument();
      expect(screen.getByText(/clear the current selection/i)).toBeInTheDocument();
    });

    it('lists ? for toggling overlay', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('?')).toBeInTheDocument();
      expect(screen.getByText(/toggle this Board Controls/i)).toBeInTheDocument();
    });

    it('lists M for sound toggle', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('M')).toBeInTheDocument();
      expect(screen.getByText(/toggle sound\/mute/i)).toBeInTheDocument();
    });

    it('lists F for fullscreen', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('F')).toBeInTheDocument();
      expect(screen.getByText(/toggle fullscreen/i)).toBeInTheDocument();
    });

    it('lists Tab for decision dialog navigation', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByText('Tab')).toBeInTheDocument();
      expect(screen.getByText(/in decision dialogs, move between options/i)).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has proper heading hierarchy', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      const h2 = screen.getByRole('heading', { level: 2 });
      expect(h2).toHaveTextContent('Board controls & shortcuts');

      const h3s = screen.getAllByRole('heading', { level: 3 });
      expect(h3s.length).toBeGreaterThanOrEqual(2);
    });

    it('has labeled sections', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByRole('region', { name: /Basic board controls/i })).toBeInTheDocument();
      expect(screen.getByRole('region', { name: /Keyboard shortcuts/i })).toBeInTheDocument();
    });

    it('close button has accessible label', () => {
      render(<BoardControlsOverlay {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Close board controls/i })).toBeInTheDocument();
    });
  });
});
