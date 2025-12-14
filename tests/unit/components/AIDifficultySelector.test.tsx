import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import {
  AIDifficultySelector,
  AIDifficultyBadge,
} from '../../../src/client/components/AIDifficultySelector';

describe('AIDifficultySelector', () => {
  const defaultProps = {
    value: 5,
    onChange: jest.fn(),
    playerNumber: 1,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('standard mode rendering', () => {
    it('renders slider with correct value', () => {
      render(<AIDifficultySelector {...defaultProps} />);

      const slider = screen.getByRole('slider');
      expect(slider).toHaveValue('5');
      expect(slider).toHaveAttribute('aria-valuenow', '5');
    });

    it('renders AI Strength label', () => {
      render(<AIDifficultySelector {...defaultProps} />);

      expect(screen.getByText('AI Strength')).toBeInTheDocument();
    });

    it('renders quick select buttons for key tiers (D2, D4, D6, D8)', () => {
      render(<AIDifficultySelector {...defaultProps} />);

      expect(screen.getByRole('button', { name: /D2/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /D4/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /D6/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /D8/i })).toBeInTheDocument();
    });

    it('marks selected quick button as pressed', () => {
      render(<AIDifficultySelector {...defaultProps} value={4} />);

      const d4Button = screen.getByRole('button', { name: /D4/i });
      expect(d4Button).toHaveAttribute('aria-pressed', 'true');

      const d2Button = screen.getByRole('button', { name: /D2/i });
      expect(d2Button).toHaveAttribute('aria-pressed', 'false');
    });

    it('renders difficulty description panel', () => {
      render(<AIDifficultySelector {...defaultProps} value={5} />);

      // D5 badge should be visible
      expect(screen.getByText('D5')).toBeInTheDocument();
    });

    it('displays difficulty name in description panel', () => {
      render(<AIDifficultySelector {...defaultProps} value={1} />);

      expect(screen.getByText('D1')).toBeInTheDocument();
    });

    it('has live region for screen reader announcements', () => {
      render(<AIDifficultySelector {...defaultProps} />);

      const liveRegion = screen.getByRole('slider').getAttribute('aria-describedby');
      expect(liveRegion).toBeTruthy();

      const descriptionElement = document.getElementById(liveRegion!);
      expect(descriptionElement).toHaveAttribute('aria-live', 'polite');
      expect(descriptionElement).toHaveAttribute('aria-atomic', 'true');
    });
  });

  describe('slider interaction', () => {
    it('calls onChange when slider value changes', async () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: '7' } });

      expect(onChange).toHaveBeenCalledWith(7);
    });

    it('has correct min/max attributes', () => {
      render(<AIDifficultySelector {...defaultProps} />);

      const slider = screen.getByRole('slider');
      expect(slider).toHaveAttribute('min', '1');
      expect(slider).toHaveAttribute('max', '10');
      expect(slider).toHaveAttribute('aria-valuemin', '1');
      expect(slider).toHaveAttribute('aria-valuemax', '10');
    });

    it('has aria-valuetext with difficulty name', () => {
      render(<AIDifficultySelector {...defaultProps} value={3} />);

      const slider = screen.getByRole('slider');
      expect(slider).toHaveAttribute('aria-valuetext');
      expect(slider.getAttribute('aria-valuetext')).toContain('Difficulty 3');
    });
  });

  describe('keyboard navigation', () => {
    it('decreases value on ArrowLeft', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={5} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'ArrowLeft' });

      expect(onChange).toHaveBeenCalledWith(4);
    });

    it('decreases value on ArrowDown', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={5} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'ArrowDown' });

      expect(onChange).toHaveBeenCalledWith(4);
    });

    it('increases value on ArrowRight', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={5} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'ArrowRight' });

      expect(onChange).toHaveBeenCalledWith(6);
    });

    it('increases value on ArrowUp', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={5} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'ArrowUp' });

      expect(onChange).toHaveBeenCalledWith(6);
    });

    it('jumps to minimum on Home key', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={5} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'Home' });

      expect(onChange).toHaveBeenCalledWith(1);
    });

    it('jumps to maximum on End key', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={5} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'End' });

      expect(onChange).toHaveBeenCalledWith(10);
    });

    it('does not go below minimum (1)', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={1} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'ArrowLeft' });

      expect(onChange).not.toHaveBeenCalled();
    });

    it('does not go above maximum (10)', () => {
      const onChange = jest.fn();
      render(<AIDifficultySelector {...defaultProps} value={10} onChange={onChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.keyDown(slider, { key: 'ArrowRight' });

      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe('quick select buttons', () => {
    it('calls onChange when quick select button is clicked', async () => {
      const onChange = jest.fn();
      const user = userEvent.setup();
      render(<AIDifficultySelector {...defaultProps} value={1} onChange={onChange} />);

      await user.click(screen.getByRole('button', { name: /D6/i }));

      expect(onChange).toHaveBeenCalledWith(6);
    });

    it('quick buttons are in a group with proper aria-label', () => {
      render(<AIDifficultySelector {...defaultProps} />);

      expect(screen.getByRole('group', { name: /Quick difficulty presets/i })).toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('disables slider when disabled prop is true', () => {
      render(<AIDifficultySelector {...defaultProps} disabled={true} />);

      expect(screen.getByRole('slider')).toBeDisabled();
    });

    it('disables quick select buttons when disabled', () => {
      render(<AIDifficultySelector {...defaultProps} disabled={true} />);

      expect(screen.getByRole('button', { name: /D2/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /D4/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /D6/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /D8/i })).toBeDisabled();
    });

    it('applies disabled styling', () => {
      render(<AIDifficultySelector {...defaultProps} disabled={true} />);

      const slider = screen.getByRole('slider');
      expect(slider).toHaveClass('opacity-50');
    });
  });

  describe('compact mode', () => {
    it('renders select dropdown instead of slider', () => {
      render(<AIDifficultySelector {...defaultProps} compact={true} />);

      expect(screen.getByRole('combobox')).toBeInTheDocument();
      expect(screen.queryByRole('slider')).not.toBeInTheDocument();
    });

    it('has correct value in select', () => {
      render(<AIDifficultySelector {...defaultProps} value={7} compact={true} />);

      expect(screen.getByRole('combobox')).toHaveValue('7');
    });

    it('calls onChange when select value changes', async () => {
      const onChange = jest.fn();
      const user = userEvent.setup();
      render(<AIDifficultySelector {...defaultProps} onChange={onChange} compact={true} />);

      await user.selectOptions(screen.getByRole('combobox'), '8');

      expect(onChange).toHaveBeenCalledWith(8);
    });

    it('renders all difficulty options in select', () => {
      render(<AIDifficultySelector {...defaultProps} compact={true} />);

      const options = screen.getAllByRole('option');
      expect(options).toHaveLength(10);
    });

    it('disables select when disabled', () => {
      render(<AIDifficultySelector {...defaultProps} compact={true} disabled={true} />);

      expect(screen.getByRole('combobox')).toBeDisabled();
    });

    it('has screen reader label for player number', () => {
      render(<AIDifficultySelector {...defaultProps} playerNumber={2} compact={true} />);

      expect(screen.getByLabelText(/AI Difficulty for Player 2/i)).toBeInTheDocument();
    });

    it('does not render quick select buttons', () => {
      render(<AIDifficultySelector {...defaultProps} compact={true} />);

      expect(screen.queryByRole('button', { name: /D2/i })).not.toBeInTheDocument();
    });
  });
});

describe('AIDifficultyBadge', () => {
  const defaultProps = {
    value: 5,
    onChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders select dropdown with current value', () => {
    render(<AIDifficultyBadge {...defaultProps} />);

    expect(screen.getByRole('combobox')).toHaveValue('5');
  });

  it('renders all difficulty options', () => {
    render(<AIDifficultyBadge {...defaultProps} />);

    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(10);
  });

  it('calls onChange when value changes', async () => {
    const onChange = jest.fn();
    const user = userEvent.setup();
    render(<AIDifficultyBadge {...defaultProps} onChange={onChange} />);

    await user.selectOptions(screen.getByRole('combobox'), '3');

    expect(onChange).toHaveBeenCalledWith(3);
  });

  it('has aria-label with difficulty information', () => {
    render(<AIDifficultyBadge {...defaultProps} value={6} />);

    const select = screen.getByRole('combobox');
    expect(select).toHaveAttribute('aria-label');
    expect(select.getAttribute('aria-label')).toContain('Difficulty');
  });

  it('disables select when disabled prop is true', () => {
    render(<AIDifficultyBadge {...defaultProps} disabled={true} />);

    expect(screen.getByRole('combobox')).toBeDisabled();
  });

  it('applies disabled styling', () => {
    render(<AIDifficultyBadge {...defaultProps} disabled={true} />);

    const select = screen.getByRole('combobox');
    expect(select).toHaveClass('opacity-50');
  });

  it('has screen reader label', () => {
    render(<AIDifficultyBadge {...defaultProps} />);

    expect(screen.getByLabelText(/AI Difficulty Level/i)).toBeInTheDocument();
  });

  it('displays D prefix in options', () => {
    render(<AIDifficultyBadge {...defaultProps} />);

    const options = screen.getAllByRole('option');
    expect(options[0]).toHaveTextContent('D1');
    expect(options[9]).toHaveTextContent('D10');
  });
});
