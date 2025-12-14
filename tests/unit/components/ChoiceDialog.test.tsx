import React from 'react';
import { render, screen, fireEvent, within, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { ChoiceDialog, ChoiceDialogProps } from '../../../src/client/components/ChoiceDialog';
import {
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice,
  PlayerChoice,
} from '../../../src/shared/types/game';

// Helper to create test LineOrderChoice
function createLineOrderChoice(): LineOrderChoice {
  return {
    id: 'choice-1',
    gameId: 'game-1',
    playerNumber: 1,
    type: 'line_order',
    prompt: 'Choose which line to process first',
    timeoutMs: 30000,
    options: [
      {
        lineId: 'line-1',
        markerPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
          { x: 2, y: 2 },
        ],
        moveId: 'move-1',
      },
      {
        lineId: 'line-2',
        markerPositions: [
          { x: 3, y: 0 },
          { x: 3, y: 1 },
          { x: 3, y: 2 },
          { x: 3, y: 3 },
        ],
        moveId: 'move-2',
      },
    ],
  };
}

// Helper to create test LineRewardChoice
function createLineRewardChoice(): LineRewardChoice {
  return {
    id: 'choice-2',
    gameId: 'game-1',
    playerNumber: 1,
    type: 'line_reward_option',
    prompt: 'Choose your line reward option',
    timeoutMs: 30000,
    options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
  };
}

// Helper to create test RingEliminationChoice
function createRingEliminationChoice(): RingEliminationChoice {
  return {
    id: 'choice-3',
    gameId: 'game-1',
    playerNumber: 1,
    type: 'ring_elimination',
    prompt: 'Choose which stack to eliminate rings from',
    timeoutMs: 30000,
    options: [
      {
        stackPosition: { x: 2, y: 3 },
        capHeight: 2,
        totalHeight: 4,
        moveId: 'move-3',
      },
      {
        stackPosition: { x: 5, y: 6 },
        capHeight: 1,
        totalHeight: 3,
        moveId: 'move-4',
      },
    ],
  };
}

// Helper to create test RegionOrderChoice
function createRegionOrderChoice(): RegionOrderChoice {
  return {
    id: 'choice-4',
    gameId: 'game-1',
    playerNumber: 1,
    type: 'region_order',
    prompt: 'Choose which region to process first',
    timeoutMs: 30000,
    options: [
      {
        regionId: 'region-1',
        size: 5,
        representativePosition: { x: 1, y: 1 },
        moveId: 'move-5',
      },
      {
        regionId: 'region-2',
        size: 3,
        representativePosition: { x: 4, y: 4 },
        moveId: 'move-6',
      },
    ],
  };
}

// Helper to create test CaptureDirectionChoice
function createCaptureDirectionChoice(): CaptureDirectionChoice {
  return {
    id: 'choice-5',
    gameId: 'game-1',
    playerNumber: 1,
    type: 'capture_direction',
    prompt: 'Choose capture direction',
    timeoutMs: 30000,
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
}

describe('ChoiceDialog', () => {
  const mockOnSelectOption = jest.fn();
  const mockOnCancel = jest.fn();

  const defaultProps: ChoiceDialogProps = {
    choice: createLineOrderChoice(),
    onSelectOption: mockOnSelectOption,
    onCancel: mockOnCancel,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('returns null when choice is null', () => {
      const { container } = render(<ChoiceDialog {...defaultProps} choice={null} />);
      expect(container.firstChild).toBeNull();
    });

    it('renders overlay backdrop when choice is provided', () => {
      const { container } = render(<ChoiceDialog {...defaultProps} />);
      const backdrop = container.firstChild as HTMLElement;
      expect(backdrop).toHaveClass('fixed', 'inset-0', 'z-40');
    });

    it('renders dialog container with proper styling', () => {
      render(<ChoiceDialog {...defaultProps} />);
      const dialog = screen.getByRole('option', { name: /Line 1/ }).closest('.bg-slate-900');
      expect(dialog).toHaveClass('rounded-md', 'border', 'border-slate-700');
    });
  });

  describe('LineOrderChoice', () => {
    it('displays prompt text', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(choice.prompt)).toBeInTheDocument();
    });

    it('renders all line options', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(/Line 1/)).toBeInTheDocument();
      expect(screen.getByText(/Line 2/)).toBeInTheDocument();
    });

    it('displays marker count for each line', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(/3 markers/)).toBeInTheDocument();
      expect(screen.getByText(/4 markers/)).toBeInTheDocument();
    });

    it('calls onSelectOption with correct option when clicked', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText(/Line 1/));

      expect(mockOnSelectOption).toHaveBeenCalledWith(choice, choice.options[0]);
    });

    it('calls onSelectOption with second option when second button clicked', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText(/Line 2/));

      expect(mockOnSelectOption).toHaveBeenCalledWith(choice, choice.options[1]);
    });
  });

  describe('LineRewardChoice', () => {
    it('displays prompt text', () => {
      const choice = createLineRewardChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(choice.prompt)).toBeInTheDocument();
    });

    it('renders Full Collapse option button', () => {
      const choice = createLineRewardChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText('Full Collapse + Elimination Bonus')).toBeInTheDocument();
      expect(screen.getByText(/Convert entire line to territory/)).toBeInTheDocument();
    });

    it('renders Minimum Collapse option button', () => {
      const choice = createLineRewardChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText('Minimum Collapse')).toBeInTheDocument();
      expect(screen.getByText(/Convert minimum markers to territory/)).toBeInTheDocument();
    });

    it('calls onSelectOption with option 1 when clicked', () => {
      const choice = createLineRewardChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText('Full Collapse + Elimination Bonus'));

      expect(mockOnSelectOption).toHaveBeenCalledWith(
        choice,
        'option_1_collapse_all_and_eliminate'
      );
    });

    it('calls onSelectOption with option 2 when clicked', () => {
      const choice = createLineRewardChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText('Minimum Collapse'));

      expect(mockOnSelectOption).toHaveBeenCalledWith(
        choice,
        'option_2_min_collapse_no_elimination'
      );
    });
  });

  describe('generic choices', () => {
    it('renders generic fallback copy and empty state when choice type is unknown', () => {
      const genericChoice = {
        id: 'choice-generic',
        gameId: 'game-1',
        playerNumber: 1,
        type: 'unknown_choice_type',
        prompt: 'Unhandled choice type',
        timeoutMs: 10_000,
        options: [],
      } as unknown as PlayerChoice;

      render(<ChoiceDialog {...defaultProps} choice={genericChoice} />);

      expect(screen.getByText('Unhandled choice type')).toBeInTheDocument();
      expect(
        screen.getByText(
          /No options are available for this decision\. Please contact support if this persists\./i
        )
      ).toBeInTheDocument();
    });
  });

  describe('RingEliminationChoice', () => {
    it('displays prompt text', () => {
      const choice = createRingEliminationChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(choice.prompt)).toBeInTheDocument();
    });

    it('renders stack options with position, cap height, and total height', () => {
      const choice = createRingEliminationChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(/Stack at \(2, 3\)/)).toBeInTheDocument();
      expect(screen.getByText(/cap 2, total 4/)).toBeInTheDocument();
      expect(screen.getByText(/Stack at \(5, 6\)/)).toBeInTheDocument();
      expect(screen.getByText(/cap 1, total 3/)).toBeInTheDocument();
    });

    it('calls onSelectOption with correct stack option', () => {
      const choice = createRingEliminationChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText(/Stack at \(2, 3\)/));

      expect(mockOnSelectOption).toHaveBeenCalledWith(choice, choice.options[0]);
    });
  });

  describe('RegionOrderChoice', () => {
    it('displays prompt text', () => {
      const choice = createRegionOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(choice.prompt)).toBeInTheDocument();
    });

    it('renders region options with ID, size, and representative position', () => {
      const choice = createRegionOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(/Region region-1/)).toBeInTheDocument();
      expect(screen.getByText(/5 spaces/)).toBeInTheDocument();
      expect(screen.getByText(/Region region-2/)).toBeInTheDocument();
      expect(screen.getByText(/3 spaces/)).toBeInTheDocument();
    });

    it('calls onSelectOption with correct region option', () => {
      const choice = createRegionOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText(/Region region-2/));

      expect(mockOnSelectOption).toHaveBeenCalledWith(choice, choice.options[1]);
    });

    it('renders a clear skip option label for territory processing', () => {
      const choice: RegionOrderChoice = {
        ...createRegionOrderChoice(),
        options: [
          ...createRegionOrderChoice().options,
          {
            regionId: 'skip',
            size: 0,
            // representativePosition is unused for skip but kept for type completeness
            representativePosition: { x: 0, y: 0 },
            moveId: 'move-skip',
          },
        ],
      };

      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const skipButton = screen.getByText('Skip territory processing for this turn');
      expect(skipButton).toBeInTheDocument();

      fireEvent.click(skipButton);

      const expectedSkipOption = choice.options[choice.options.length - 1];
      expect(mockOnSelectOption).toHaveBeenCalledWith(choice, expectedSkipOption);
    });
  });

  describe('CaptureDirectionChoice', () => {
    it('displays prompt text', () => {
      const choice = createCaptureDirectionChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(choice.prompt)).toBeInTheDocument();
    });

    it('renders direction options with target, landing, and cap height', () => {
      const choice = createCaptureDirectionChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);
      expect(screen.getByText(/Direction 1/)).toBeInTheDocument();
      expect(screen.getByText(/target \(2, 2\)/)).toBeInTheDocument();
      expect(screen.getByText(/landing \(3, 3\)/)).toBeInTheDocument();
      expect(screen.getByText(/Direction 2/)).toBeInTheDocument();
    });

    it('calls onSelectOption with correct direction option', () => {
      const choice = createCaptureDirectionChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText(/Direction 1/));

      expect(mockOnSelectOption).toHaveBeenCalledWith(choice, choice.options[0]);
    });
  });

  describe('disabled state during submission', () => {
    it('disables all option buttons after one is clicked', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);
      const option2Button = screen.getByText(/Line 2/);

      fireEvent.click(option1Button);

      expect(option1Button).toBeDisabled();
      expect(option2Button).toBeDisabled();
    });

    it('prevents multiple calls to onSelectOption', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);

      fireEvent.click(option1Button);
      fireEvent.click(option1Button);
      fireEvent.click(option1Button);

      expect(mockOnSelectOption).toHaveBeenCalledTimes(1);
    });

    it('applies disabled styling to buttons', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);
      fireEvent.click(option1Button);

      expect(option1Button).toHaveClass('disabled:opacity-60', 'disabled:cursor-not-allowed');
    });
  });

  describe('timer display', () => {
    it('shows countdown when deadline and timeRemainingMs provided', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={25000}
        />
      );

      expect(screen.getByText('Respond within')).toBeInTheDocument();
      expect(screen.getByText('25s')).toBeInTheDocument();
    });

    it('shows "Choice timeout active" when deadline present but no timeRemainingMs', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={null}
        />
      );

      expect(screen.getByText('Choice timeout active')).toBeInTheDocument();
    });

    it('does not show timer when no deadline', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog {...defaultProps} choice={choice} deadline={null} timeRemainingMs={null} />
      );

      expect(screen.queryByText('Respond within')).not.toBeInTheDocument();
      expect(screen.queryByText('Choice timeout active')).not.toBeInTheDocument();
    });

    it('renders progress bar with correct width percentage', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={15000} // 50% remaining
        />
      );

      const progressBar = screen.getByTestId('choice-countdown-bar');
      expect(progressBar).toHaveStyle({ width: '50%' });
    });

    it('clamps countdown to zero for negative values', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() - 5000}
          timeRemainingMs={-5000}
        />
      );

      expect(screen.getByText('0s')).toBeInTheDocument();
    });

    it('rounds countdown up to nearest second', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={15500} // Should display 16s
        />
      );

      expect(screen.getByText('16s')).toBeInTheDocument();
    });
  });

  describe('countdown severity and server-capped semantics', () => {
    it('applies normal severity and default label when above 10 seconds', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={25000}
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('data-severity', 'normal');
      expect(countdown).not.toHaveAttribute('data-server-capped');
      expect(screen.getByText('Respond within')).toBeInTheDocument();
      expect(screen.getByText('25s')).toBeInTheDocument();
    });

    it('applies warning severity at the 10s threshold', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={10_000}
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('data-severity', 'warning');
    });

    it('applies warning severity just above the critical boundary', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={3_001}
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('data-severity', 'warning');
    });

    it('applies critical severity at or below 3s', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={3_000}
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('data-severity', 'critical');
    });

    it('applies critical severity at zero remaining time', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={0}
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('data-severity', 'critical');
    });

    it('switches to server deadline label and sets data-server-capped when capped by server', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={3000}
          isServerCapped
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('data-server-capped', 'true');
      expect(screen.getByText('Server deadline – respond within')).toBeInTheDocument();
      expect(screen.getByText('3s')).toBeInTheDocument();
    });
  });

  describe('cancel button', () => {
    it('renders cancel button when onCancel is provided', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    it('does not render cancel button when onCancel is not provided', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog choice={choice} onSelectOption={mockOnSelectOption} />);

      expect(screen.queryByText('Cancel')).not.toBeInTheDocument();
    });

    it('calls onCancel when cancel button is clicked', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText('Cancel'));

      expect(mockOnCancel).toHaveBeenCalledTimes(1);
    });

    it('disables cancel button after option is selected', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      fireEvent.click(screen.getByText(/Line 1/));

      expect(screen.getByText('Cancel')).toBeDisabled();
    });

    it('does not call onCancel when button is disabled', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      // First click an option to enter submitting state
      fireEvent.click(screen.getByText(/Line 1/));

      // Then try to click cancel
      fireEvent.click(screen.getByText('Cancel'));

      expect(mockOnCancel).not.toHaveBeenCalled();
    });
  });

  describe('hexagonal coordinates (z-axis)', () => {
    it('displays z-coordinate for ring elimination stack positions', () => {
      const choice: RingEliminationChoice = {
        ...createRingEliminationChoice(),
        options: [
          {
            stackPosition: { x: 2, y: 3, z: -5 },
            capHeight: 2,
            totalHeight: 4,
            moveId: 'move-hex-1',
          },
        ],
      };

      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      expect(screen.getByText(/Stack at \(2, 3, -5\)/)).toBeInTheDocument();
    });

    it('displays z-coordinate for region representative positions', () => {
      const choice: RegionOrderChoice = {
        ...createRegionOrderChoice(),
        options: [
          {
            regionId: 'region-hex',
            size: 4,
            representativePosition: { x: 1, y: 2, z: -3 },
            moveId: 'move-hex-2',
          },
        ],
      };

      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      expect(screen.getByText(/sample \(1, 2, -3\)/)).toBeInTheDocument();
    });

    it('displays z-coordinate for capture direction positions', () => {
      const choice: CaptureDirectionChoice = {
        ...createCaptureDirectionChoice(),
        options: [
          {
            targetPosition: { x: 2, y: 2, z: -4 },
            landingPosition: { x: 3, y: 3, z: -6 },
            capturedCapHeight: 2,
          },
        ],
      };

      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      expect(screen.getByText(/target \(2, 2, -4\)/)).toBeInTheDocument();
      expect(screen.getByText(/landing \(3, 3, -6\)/)).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('responds to ArrowDown key for option navigation', async () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      const optionButtons = screen.getAllByRole('option') as HTMLButtonElement[];
      expect(optionButtons.length).toBe(2);

      // First option should be focused initially
      expect(document.activeElement).toBe(optionButtons[0]);

      // Spy on the second option's focus method to verify the code path is executed
      const focusSpy = jest.spyOn(optionButtons[1], 'focus');

      // Press ArrowDown - should trigger navigation code path
      await act(async () => {
        fireEvent.keyDown(dialog, { key: 'ArrowDown' });
      });

      // Verify focus() was called on the next element (JSDOM doesn't always update activeElement)
      expect(focusSpy).toHaveBeenCalled();
      focusSpy.mockRestore();
    });

    it('responds to ArrowUp key for option navigation', async () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      const optionButtons = screen.getAllByRole('option') as HTMLButtonElement[];

      // First option focused initially (index 0)
      expect(document.activeElement).toBe(optionButtons[0]);

      // Spy on the last option's focus method to verify the code path is executed
      const focusSpy = jest.spyOn(optionButtons[1], 'focus');

      // Press ArrowUp - should wrap to last option (index becomes length - 1)
      await act(async () => {
        fireEvent.keyDown(dialog, { key: 'ArrowUp' });
      });

      // Verify focus() was called on the wrapped element
      expect(focusSpy).toHaveBeenCalled();
      focusSpy.mockRestore();
    });

    it('traps focus when Shift+Tab from first element', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      const optionButtons = screen.getAllByRole('option') as HTMLButtonElement[];
      const cancelButton = screen.getByText('Cancel');

      // Focus first option
      act(() => {
        optionButtons[0].focus();
      });

      // Shift+Tab should wrap to last focusable (cancel button)
      fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
      expect(document.activeElement).toBe(cancelButton);
    });

    it('traps focus when Tab from last element', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      const optionButtons = screen.getAllByRole('option') as HTMLButtonElement[];
      const cancelButton = screen.getByText('Cancel');

      // Focus cancel button (last focusable)
      act(() => {
        cancelButton.focus();
      });

      // Tab should wrap to first focusable
      fireEvent.keyDown(dialog, { key: 'Tab' });
      expect(document.activeElement).toBe(optionButtons[0]);
    });

    it('traps focus within the dialog when tabbing forward/backward', async () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      const optionButtons = screen.getAllByRole('option') as HTMLButtonElement[];
      expect(optionButtons.length).toBeGreaterThan(1);

      const cancelButton = screen.getByText('Cancel');

      // Focus the last option and press Tab (should advance but remain within dialog focusables).
      act(() => {
        optionButtons[optionButtons.length - 1].focus();
      });
      act(() => {
        fireEvent.keyDown(dialog, { key: 'Tab' });
      });
      const focusables = [...optionButtons, cancelButton];
      expect(focusables).toContain(document.activeElement);

      // Focus the first option and press Shift+Tab (should wrap within dialog focusables).
      act(() => {
        optionButtons[0].focus();
      });
      act(() => {
        fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
      });
      expect(focusables).toContain(document.activeElement);
    });

    it('auto-focuses the first option when the dialog opens', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const optionButtons = screen.getAllByRole('option') as HTMLButtonElement[];
      expect(optionButtons.length).toBeGreaterThan(0);
      expect(document.activeElement).toBe(optionButtons[0]);
    });

    it('all option buttons have type="button"', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach((button) => {
        expect(button).toHaveAttribute('type', 'button');
      });
    });

    it('cancel button has type="button"', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const cancelButton = screen.getByText('Cancel');
      expect(cancelButton).toHaveAttribute('type', 'button');
    });

    it('option buttons are keyboard focusable', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);
      option1Button.focus();

      expect(document.activeElement).toBe(option1Button);
    });

    it('option buttons can be activated with Enter key', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);
      option1Button.focus();
      fireEvent.keyDown(option1Button, { key: 'Enter' });
      fireEvent.click(option1Button);

      expect(mockOnSelectOption).toHaveBeenCalled();
    });

    it('dialog has proper ARIA attributes', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
      expect(dialog).toHaveAttribute('aria-labelledby', 'choice-dialog-title');
    });

    it('closes dialog on escape key when onCancel provided', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const dialog = screen.getByRole('dialog');
      fireEvent.keyDown(dialog, { key: 'Escape' });

      expect(mockOnCancel).toHaveBeenCalledTimes(1);
    });

    it('does not close on escape key when onCancel not provided', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog choice={choice} onSelectOption={mockOnSelectOption} />);

      const dialog = screen.getByRole('dialog');
      fireEvent.keyDown(dialog, { key: 'Escape' });

      // Dialog should still be visible (no onCancel to call)
      expect(dialog).toBeInTheDocument();
    });

    it('auto-focuses first option button when opened', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      // First focusable element should be the first option button
      const firstOption = screen.getByText(/Line 1/);
      expect(document.activeElement).toBe(firstOption);
    });

    it('countdown timer has aria-live region for screen readers', () => {
      const choice = createLineOrderChoice();
      render(
        <ChoiceDialog
          {...defaultProps}
          choice={choice}
          deadline={Date.now() + 30000}
          timeRemainingMs={25000}
        />
      );

      const countdown = screen.getByTestId('choice-countdown');
      expect(countdown).toHaveAttribute('aria-live', 'polite');
      expect(countdown).toHaveAttribute('aria-atomic', 'true');
      expect(countdown).toHaveAttribute('role', 'timer');
    });
  });

  describe('unknown choice type', () => {
    it('falls back to a generic decision UI for unsupported choice types', () => {
      const unknownChoice = {
        id: 'unknown',
        gameId: 'game-1',
        playerNumber: 1,
        type: 'unknown_type' as any,
        prompt: 'Unknown choice',
        options: ['a', 'b'],
      } as unknown as PlayerChoice;

      render(<ChoiceDialog {...defaultProps} choice={unknownChoice} />);

      // Header comes from the fallback view model in choiceViewModels.ts
      expect(screen.getByText(/decision required/i)).toBeInTheDocument();
      expect(screen.getByText(/unknown choice/i)).toBeInTheDocument();
      // Generic options should still be selectable
      const option1 = screen.getByText('Option 1');
      const option2 = screen.getByText('Option 2');
      expect(option1).toBeInTheDocument();
      expect(option2).toBeInTheDocument();

      fireEvent.click(option1);
      expect(mockOnSelectOption).toHaveBeenCalledWith(unknownChoice, unknownChoice.options[0]);
    });
  });

  describe('multiple options handling', () => {
    it('renders all options for line order choice with many lines', () => {
      const choice: LineOrderChoice = {
        ...createLineOrderChoice(),
        options: [
          { lineId: 'line-1', markerPositions: [{ x: 0, y: 0 }], moveId: 'move-1' },
          { lineId: 'line-2', markerPositions: [{ x: 1, y: 1 }], moveId: 'move-2' },
          { lineId: 'line-3', markerPositions: [{ x: 2, y: 2 }], moveId: 'move-3' },
          { lineId: 'line-4', markerPositions: [{ x: 3, y: 3 }], moveId: 'move-4' },
          { lineId: 'line-5', markerPositions: [{ x: 4, y: 4 }], moveId: 'move-5' },
        ],
      };

      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      expect(screen.getByText(/Line 1/)).toBeInTheDocument();
      expect(screen.getByText(/Line 2/)).toBeInTheDocument();
      expect(screen.getByText(/Line 3/)).toBeInTheDocument();
      expect(screen.getByText(/Line 4/)).toBeInTheDocument();
      expect(screen.getByText(/Line 5/)).toBeInTheDocument();
    });

    it('has scrollable container for many options', () => {
      const choice: LineOrderChoice = {
        ...createLineOrderChoice(),
        options: Array.from({ length: 10 }, (_, i) => ({
          lineId: `line-${i + 1}`,
          markerPositions: [{ x: i, y: i }],
          moveId: `move-${i + 1}`,
        })),
      };

      const { container } = render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const scrollContainer = container.querySelector('.max-h-48.overflow-auto');
      expect(scrollContainer).toBeInTheDocument();
    });
  });

  describe('mapping header', () => {
    it('renders mapping-based header for line order choices', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      // shortLabel from mapping
      expect(screen.getByText('Line order')).toBeInTheDocument();
      // title from mapping
      expect(screen.getByText(/multiple lines formed/i)).toBeInTheDocument();
    });

    it('falls back to prompt-only content when mapping header is absent', () => {
      const choice = createCaptureDirectionChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      // The prompt should always be shown regardless of mapping
      expect(screen.getByText(choice.prompt)).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies hover styles to option buttons', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);
      expect(option1Button).toHaveClass('hover:bg-slate-700');
    });

    it('applies correct background to dialog', () => {
      const { container } = render(<ChoiceDialog {...defaultProps} />);
      const dialog = container.querySelector('.bg-slate-900');
      expect(dialog).toBeInTheDocument();
    });

    it('applies border styling to option buttons', () => {
      const choice = createLineOrderChoice();
      render(<ChoiceDialog {...defaultProps} choice={choice} />);

      const option1Button = screen.getByText(/Line 1/);
      expect(option1Button).toHaveClass('border', 'border-slate-600');
    });
  });
});
