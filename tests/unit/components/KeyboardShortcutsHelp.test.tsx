import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { KeyboardShortcutsHelp } from '../../../src/client/components/KeyboardShortcutsHelp';

describe('KeyboardShortcutsHelp', () => {
  const mockOnClose = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('visibility', () => {
    it('should not render when isOpen is false', () => {
      render(<KeyboardShortcutsHelp isOpen={false} onClose={mockOnClose} />);

      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('should render when isOpen is true', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });
  });

  describe('content', () => {
    it('should display the title', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument();
    });

    it('should display Board Navigation section', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('Board Navigation')).toBeInTheDocument();
      expect(screen.getByText('Navigate between board cells')).toBeInTheDocument();
    });

    it('should display Dialog Navigation section', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('Dialog Navigation')).toBeInTheDocument();
      expect(screen.getByText('Navigate between options')).toBeInTheDocument();
    });

    it('should display General section', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('General')).toBeInTheDocument();
      expect(screen.getByText('Navigate to next interactive element')).toBeInTheDocument();
    });

    it('should display shortcut keys', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Check for arrow keys (multiple instances exist)
      expect(screen.getAllByText('↑').length).toBeGreaterThan(0);
      expect(screen.getAllByText('↓').length).toBeGreaterThan(0);
      expect(screen.getAllByText('←').length).toBeGreaterThan(0);
      expect(screen.getAllByText('→').length).toBeGreaterThan(0);

      // Check for other keys
      expect(screen.getAllByText('Enter').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Space').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Escape').length).toBeGreaterThan(0);
    });
  });

  describe('close button', () => {
    it('should call onClose when close button is clicked', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', { name: /close keyboard shortcuts help/i });
      fireEvent.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('backdrop click', () => {
    it('should call onClose when clicking the backdrop', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Get the backdrop (the outer div with role="presentation")
      const backdrop = screen.getByRole('presentation');
      fireEvent.click(backdrop);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should not call onClose when clicking inside the dialog', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const dialog = screen.getByRole('dialog');
      fireEvent.click(dialog);

      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('keyboard interactions', () => {
    it('should call onClose when Escape key is pressed', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      fireEvent.keyDown(document, { key: 'Escape' });

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when ? key is pressed', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      fireEvent.keyDown(document, { key: '?' });

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('accessibility', () => {
    it('should have aria-modal attribute', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
    });

    it('should have aria-labelledby pointing to title', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-labelledby', 'keyboard-shortcuts-title');

      const title = document.getElementById('keyboard-shortcuts-title');
      expect(title).toHaveTextContent('Keyboard Shortcuts');
    });

    it('should have aria-label on close button', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', { name: /close keyboard shortcuts help/i });
      expect(closeButton).toHaveAttribute('aria-label', 'Close keyboard shortcuts help');
    });
  });

  describe('shortcut sections', () => {
    it('should display all board shortcuts', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('Navigate between board cells')).toBeInTheDocument();
      expect(screen.getByText('Select current cell')).toBeInTheDocument();
      expect(screen.getByText('Cancel current action / Clear selection')).toBeInTheDocument();
      expect(screen.getByText('Show this help dialog')).toBeInTheDocument();
    });

    it('should display all dialog shortcuts', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('Navigate between options')).toBeInTheDocument();
      expect(screen.getByText('Select focused option')).toBeInTheDocument();
      expect(screen.getByText('Move focus between elements')).toBeInTheDocument();
      expect(screen.getByText('Close dialog (if cancellable)')).toBeInTheDocument();
    });

    it('should display all general shortcuts', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByText('Navigate to next interactive element')).toBeInTheDocument();
      expect(screen.getByText('Navigate to previous interactive element')).toBeInTheDocument();
    });
  });
});
