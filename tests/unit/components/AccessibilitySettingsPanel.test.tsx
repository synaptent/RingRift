import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { AccessibilitySettingsPanel } from '../../../src/client/components/AccessibilitySettingsPanel';
import * as AccessibilityContext from '../../../src/client/contexts/AccessibilityContext';

// Mock the useAccessibility hook
const mockSetPreference = jest.fn();
const mockResetPreferences = jest.fn();
const mockGetPlayerColor = jest.fn(
  (index: number) => ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'][index]
);

jest.mock('../../../src/client/contexts/AccessibilityContext', () => ({
  ...jest.requireActual('../../../src/client/contexts/AccessibilityContext'),
  useAccessibility: jest.fn(),
}));

const mockUseAccessibility = AccessibilityContext.useAccessibility as jest.Mock;

describe('AccessibilitySettingsPanel', () => {
  const defaultAccessibilityState = {
    highContrastMode: false,
    colorVisionMode: 'normal' as AccessibilityContext.ColorVisionMode,
    reducedMotion: false,
    largeText: false,
    systemPrefersReducedMotion: false,
    setPreference: mockSetPreference,
    resetPreferences: mockResetPreferences,
    getPlayerColor: mockGetPlayerColor,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseAccessibility.mockReturnValue(defaultAccessibilityState);
  });

  describe('rendering', () => {
    it('renders the panel heading', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByRole('heading', { name: 'Accessibility' })).toBeInTheDocument();
    });

    it('renders reset to defaults button', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByRole('button', { name: /Reset to defaults/i })).toBeInTheDocument();
    });

    it('renders High Contrast Mode toggle', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/High Contrast Mode/i)).toBeInTheDocument();
      expect(screen.getByText(/Increases visual distinction/i)).toBeInTheDocument();
    });

    it('renders Reduce Motion toggle', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/Reduce Motion/i)).toBeInTheDocument();
      expect(screen.getByText(/Disables non-essential animations/i)).toBeInTheDocument();
    });

    it('renders Large Text toggle', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/Large Text/i)).toBeInTheDocument();
      expect(screen.getByText(/Increases base font sizes/i)).toBeInTheDocument();
    });

    it('renders Color Vision Mode select', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/Color Vision Mode/i)).toBeInTheDocument();
      expect(screen.getByText(/Select a color palette/i)).toBeInTheDocument();
    });

    it('renders color preview section', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByText(/Player color preview/i)).toBeInTheDocument();
      expect(screen.getByTitle('Player 1')).toBeInTheDocument();
      expect(screen.getByTitle('Player 2')).toBeInTheDocument();
      expect(screen.getByTitle('Player 3')).toBeInTheDocument();
      expect(screen.getByTitle('Player 4')).toBeInTheDocument();
    });

    it('renders keyboard shortcuts hint', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByText(/keyboard shortcuts/i)).toBeInTheDocument();
    });
  });

  describe('toggle states', () => {
    it('shows High Contrast toggle as unchecked when false', () => {
      render(<AccessibilitySettingsPanel />);

      const toggle = screen.getByLabelText(/High Contrast Mode/i);
      expect(toggle).toHaveAttribute('aria-checked', 'false');
    });

    it('shows High Contrast toggle as checked when true', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        highContrastMode: true,
      });
      render(<AccessibilitySettingsPanel />);

      const toggle = screen.getByLabelText(/High Contrast Mode/i);
      expect(toggle).toHaveAttribute('aria-checked', 'true');
    });

    it('shows Reduce Motion toggle as checked when reducedMotion is true', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        reducedMotion: true,
      });
      render(<AccessibilitySettingsPanel />);

      const toggle = screen.getByLabelText(/Reduce Motion/i);
      expect(toggle).toHaveAttribute('aria-checked', 'true');
    });

    it('shows Reduce Motion toggle as checked when system prefers reduced motion', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        systemPrefersReducedMotion: true,
      });
      render(<AccessibilitySettingsPanel />);

      const toggle = screen.getByLabelText(/Reduce Motion/i);
      expect(toggle).toHaveAttribute('aria-checked', 'true');
    });

    it('shows Large Text toggle state correctly', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        largeText: true,
      });
      render(<AccessibilitySettingsPanel />);

      const toggle = screen.getByLabelText(/Large Text/i);
      expect(toggle).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('interactions', () => {
    it('calls setPreference when High Contrast toggle is clicked', async () => {
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel />);

      await user.click(screen.getByLabelText(/High Contrast Mode/i));

      expect(mockSetPreference).toHaveBeenCalledWith('highContrastMode', true);
    });

    it('calls setPreference when Reduce Motion toggle is clicked', async () => {
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel />);

      await user.click(screen.getByLabelText(/Reduce Motion/i));

      expect(mockSetPreference).toHaveBeenCalledWith('reducedMotion', true);
    });

    it('calls setPreference when Large Text toggle is clicked', async () => {
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel />);

      await user.click(screen.getByLabelText(/Large Text/i));

      expect(mockSetPreference).toHaveBeenCalledWith('largeText', true);
    });

    it('calls setPreference when Color Vision Mode select changes', async () => {
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel />);

      await user.selectOptions(screen.getByLabelText(/Color Vision Mode/i), 'deuteranopia');

      expect(mockSetPreference).toHaveBeenCalledWith('colorVisionMode', 'deuteranopia');
    });

    it('calls resetPreferences when reset button is clicked', async () => {
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel />);

      await user.click(screen.getByRole('button', { name: /Reset to defaults/i }));

      expect(mockResetPreferences).toHaveBeenCalled();
    });

    it('calls onSettingsChange callback when a setting changes', async () => {
      const onSettingsChange = jest.fn();
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel onSettingsChange={onSettingsChange} />);

      await user.click(screen.getByLabelText(/High Contrast Mode/i));

      expect(onSettingsChange).toHaveBeenCalled();
    });

    it('calls onSettingsChange callback when reset is clicked', async () => {
      const onSettingsChange = jest.fn();
      const user = userEvent.setup();
      render(<AccessibilitySettingsPanel onSettingsChange={onSettingsChange} />);

      await user.click(screen.getByRole('button', { name: /Reset to defaults/i }));

      expect(onSettingsChange).toHaveBeenCalled();
    });
  });

  describe('color vision options', () => {
    it('has all color vision mode options', () => {
      render(<AccessibilitySettingsPanel />);

      const select = screen.getByLabelText(/Color Vision Mode/i);
      const options = select.querySelectorAll('option');

      expect(options).toHaveLength(4);
      expect(options[0]).toHaveValue('normal');
      expect(options[1]).toHaveValue('deuteranopia');
      expect(options[2]).toHaveValue('protanopia');
      expect(options[3]).toHaveValue('tritanopia');
    });

    it('shows current color vision mode as selected', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        colorVisionMode: 'protanopia',
      });
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/Color Vision Mode/i)).toHaveValue('protanopia');
    });
  });

  describe('system preference detection', () => {
    it('shows system preference warning when system prefers reduced motion but user setting is off', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        systemPrefersReducedMotion: true,
        reducedMotion: false,
      });
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByText(/System reduced motion is enabled/i)).toBeInTheDocument();
    });

    it('does not show warning when both system and user prefer reduced motion', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        systemPrefersReducedMotion: true,
        reducedMotion: true,
      });
      render(<AccessibilitySettingsPanel />);

      expect(screen.queryByText(/System reduced motion is enabled/i)).not.toBeInTheDocument();
    });

    it('shows system preference in description when system prefers reduced motion', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        systemPrefersReducedMotion: true,
      });
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByText(/Your system prefers reduced motion/i)).toBeInTheDocument();
    });
  });

  describe('compact mode', () => {
    it('shows shorter descriptions in compact mode', () => {
      render(<AccessibilitySettingsPanel compact={true} />);

      expect(screen.getByText('Stronger borders and colors')).toBeInTheDocument();
      expect(screen.queryByText(/Increases visual distinction/i)).not.toBeInTheDocument();
    });

    it('shows compact Large Text description', () => {
      render(<AccessibilitySettingsPanel compact={true} />);

      expect(screen.getByText('Increase font sizes')).toBeInTheDocument();
      expect(
        screen.queryByText(/Increases base font sizes by approximately 25%/i)
      ).not.toBeInTheDocument();
    });

    it('shows compact Color Vision description', () => {
      render(<AccessibilitySettingsPanel compact={true} />);

      expect(screen.getByText('Adjust colors for color blindness')).toBeInTheDocument();
      expect(screen.queryByText(/Select a color palette optimized/i)).not.toBeInTheDocument();
    });

    it('shows compact system preference description', () => {
      mockUseAccessibility.mockReturnValue({
        ...defaultAccessibilityState,
        systemPrefersReducedMotion: true,
      });
      render(<AccessibilitySettingsPanel compact={true} />);

      expect(screen.getByText('System preference detected')).toBeInTheDocument();
    });
  });

  describe('color preview', () => {
    it('displays player colors from getPlayerColor', () => {
      render(<AccessibilitySettingsPanel />);

      const player1 = screen.getByTitle('Player 1');
      expect(player1).toHaveStyle({ backgroundColor: '#10b981' });

      const player2 = screen.getByTitle('Player 2');
      expect(player2).toHaveStyle({ backgroundColor: '#3b82f6' });
    });

    it('calls getPlayerColor for all 4 players', () => {
      render(<AccessibilitySettingsPanel />);

      expect(mockGetPlayerColor).toHaveBeenCalledWith(0);
      expect(mockGetPlayerColor).toHaveBeenCalledWith(1);
      expect(mockGetPlayerColor).toHaveBeenCalledWith(2);
      expect(mockGetPlayerColor).toHaveBeenCalledWith(3);
    });

    it('shows player numbers in color circles', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByTitle('Player 1')).toHaveTextContent('1');
      expect(screen.getByTitle('Player 2')).toHaveTextContent('2');
      expect(screen.getByTitle('Player 3')).toHaveTextContent('3');
      expect(screen.getByTitle('Player 4')).toHaveTextContent('4');
    });
  });

  describe('accessibility', () => {
    it('all toggles have switch role', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/High Contrast Mode/i)).toHaveAttribute('role', 'switch');
      expect(screen.getByLabelText(/Reduce Motion/i)).toHaveAttribute('role', 'switch');
      expect(screen.getByLabelText(/Large Text/i)).toHaveAttribute('role', 'switch');
    });

    it('all toggles have aria-checked attribute', () => {
      render(<AccessibilitySettingsPanel />);

      expect(screen.getByLabelText(/High Contrast Mode/i)).toHaveAttribute('aria-checked');
      expect(screen.getByLabelText(/Reduce Motion/i)).toHaveAttribute('aria-checked');
      expect(screen.getByLabelText(/Large Text/i)).toHaveAttribute('aria-checked');
    });

    it('labels are associated with inputs', () => {
      render(<AccessibilitySettingsPanel />);

      // Verify labels are clickable and associated with toggles
      expect(screen.getByLabelText(/High Contrast Mode/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Reduce Motion/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Large Text/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Color Vision Mode/i)).toBeInTheDocument();
    });
  });
});
