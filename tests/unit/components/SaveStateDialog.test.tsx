import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { SaveStateDialog } from '../../../src/client/components/SaveStateDialog';
import type { GameState } from '../../../src/shared/types/game';
import * as statePersistence from '../../../src/client/sandbox/statePersistence';

// Mock statePersistence module
jest.mock('../../../src/client/sandbox/statePersistence', () => ({
  saveCurrentGameState: jest.fn(),
  exportScenarioToFile: jest.fn(),
}));

const mockSaveCurrentGameState = statePersistence.saveCurrentGameState as jest.Mock;
const mockExportScenarioToFile = statePersistence.exportScenarioToFile as jest.Mock;

describe('SaveStateDialog', () => {
  const mockOnClose = jest.fn();
  const mockOnSaved = jest.fn();
  const mockGameState: GameState = {
    id: 'test-game',
    boardType: 'square8',
    phase: 'playing',
    currentPlayer: 1,
    gameStatus: 'active',
    players: [],
    board: {
      cells: [],
      boardType: 'square8',
    },
    history: [],
    availableMoves: [],
    scores: { 1: 0, 2: 0 },
    capturedRings: { 1: 0, 2: 0 },
  } as GameState;

  beforeEach(() => {
    jest.clearAllMocks();
    mockSaveCurrentGameState.mockReturnValue({
      id: 'saved-1',
      name: 'Test Save',
      description: '',
      category: 'custom',
      boardType: 'square8',
      playerCount: 2,
      tags: [],
      gameState: mockGameState,
    });
  });

  describe('visibility', () => {
    it('should not render when isOpen is false', () => {
      render(<SaveStateDialog isOpen={false} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('should render when isOpen is true', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });
  });

  describe('content', () => {
    it('should display the title', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByText('Save Game State')).toBeInTheDocument();
    });

    it('should display name input field', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByLabelText(/Name/)).toBeInTheDocument();
      expect(screen.getByPlaceholderText('My saved game')).toBeInTheDocument();
    });

    it('should display description textarea', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByLabelText(/Description/)).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Describe this game state...')).toBeInTheDocument();
    });

    it('should display export checkbox', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByText('Also export as JSON file')).toBeInTheDocument();
      expect(screen.getByRole('checkbox')).toBeInTheDocument();
    });

    it('should display Cancel and Save buttons', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
    });
  });

  describe('form interaction', () => {
    it('should allow typing in name field', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'My Test Save');

      expect(nameInput).toHaveValue('My Test Save');
    });

    it('should allow typing in description field', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const descInput = screen.getByLabelText(/Description/);
      await user.type(descInput, 'Test description');

      expect(descInput).toHaveValue('Test description');
    });

    it('should allow toggling export checkbox', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).not.toBeChecked();

      await user.click(checkbox);
      expect(checkbox).toBeChecked();
    });

    it('should disable Save button when name is empty', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const saveButton = screen.getByRole('button', { name: 'Save' });
      expect(saveButton).toBeDisabled();
    });

    it('should enable Save button when name is entered', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Valid Name');

      const saveButton = screen.getByRole('button', { name: 'Save' });
      expect(saveButton).not.toBeDisabled();
    });
  });

  describe('saving', () => {
    it('should call saveCurrentGameState on save', async () => {
      const user = userEvent.setup();
      render(
        <SaveStateDialog
          isOpen={true}
          onClose={mockOnClose}
          gameState={mockGameState}
          onSaved={mockOnSaved}
        />
      );

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      const saveButton = screen.getByRole('button', { name: 'Save' });
      await user.click(saveButton);

      expect(mockSaveCurrentGameState).toHaveBeenCalledWith(mockGameState, {
        name: 'Test Save',
        description: undefined,
      });
    });

    it('should call onSaved callback after saving', async () => {
      const user = userEvent.setup();
      render(
        <SaveStateDialog
          isOpen={true}
          onClose={mockOnClose}
          gameState={mockGameState}
          onSaved={mockOnSaved}
        />
      );

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      const saveButton = screen.getByRole('button', { name: 'Save' });
      await user.click(saveButton);

      expect(mockOnSaved).toHaveBeenCalled();
    });

    it('should call onClose after saving', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      const saveButton = screen.getByRole('button', { name: 'Save' });
      await user.click(saveButton);

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('should export to file when checkbox is checked', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      const checkbox = screen.getByRole('checkbox');
      await user.click(checkbox);

      const saveButton = screen.getByRole('button', { name: 'Save' });
      await user.click(saveButton);

      expect(mockExportScenarioToFile).toHaveBeenCalled();
    });

    it('should not export to file when checkbox is unchecked', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      const saveButton = screen.getByRole('button', { name: 'Save' });
      await user.click(saveButton);

      expect(mockExportScenarioToFile).not.toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('should show error when no game state available', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={null} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      // Form submit
      fireEvent.submit(nameInput.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText('No game state available to save')).toBeInTheDocument();
      });
    });

    it('should show error when name is empty on submit', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, '   '); // whitespace only

      // Form submit via Enter key
      fireEvent.submit(nameInput.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText('Please enter a name for the saved state')).toBeInTheDocument();
      });
    });

    it('should show error when saveCurrentGameState throws', async () => {
      mockSaveCurrentGameState.mockImplementation(() => {
        throw new Error('Storage quota exceeded');
      });

      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Test Save');

      const saveButton = screen.getByRole('button', { name: 'Save' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(screen.getByText('Storage quota exceeded')).toBeInTheDocument();
      });
    });
  });

  describe('cancel button', () => {
    it('should call onClose when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      await user.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('keyboard interactions', () => {
    it('should close on Escape key press', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      // The keydown handler is on the inner div with the form, not the dialog
      const dialogContent = screen.getByRole('dialog').querySelector('div')!;
      fireEvent.keyDown(dialogContent, { key: 'Escape' });

      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('accessibility', () => {
    it('should have aria-modal attribute', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
    });

    it('should have aria-labelledby pointing to title', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-labelledby', 'save-state-title');

      const title = document.getElementById('save-state-title');
      expect(title).toHaveTextContent('Save Game State');
    });

    it('should focus name input when dialog opens', () => {
      render(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      const nameInput = screen.getByLabelText(/Name/);
      expect(document.activeElement).toBe(nameInput);
    });
  });

  describe('form reset on open', () => {
    it('should reset form fields when dialog opens', async () => {
      const { rerender } = render(
        <SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />
      );

      const user = userEvent.setup();
      const nameInput = screen.getByLabelText(/Name/);
      await user.type(nameInput, 'Some name');

      // Close and reopen
      rerender(<SaveStateDialog isOpen={false} onClose={mockOnClose} gameState={mockGameState} />);

      rerender(<SaveStateDialog isOpen={true} onClose={mockOnClose} gameState={mockGameState} />);

      expect(screen.getByLabelText(/Name/)).toHaveValue('');
    });
  });
});
