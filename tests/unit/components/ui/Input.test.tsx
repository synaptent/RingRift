import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Input, InputSize } from '../../../../src/client/components/ui/Input';

describe('Input', () => {
  describe('rendering', () => {
    it('renders without crashing', () => {
      render(<Input aria-label="test input" />);
      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    it('renders with displayName set correctly', () => {
      expect(Input.displayName).toBe('Input');
    });
  });

  describe('sizes', () => {
    const sizes: InputSize[] = ['sm', 'md', 'lg'];

    it.each(sizes)('renders %s size correctly', (size) => {
      render(<Input size={size} aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toBeInTheDocument();
    });

    it('applies medium size classes by default', () => {
      render(<Input aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('text-sm', 'py-2');
    });

    it('applies small size classes', () => {
      render(<Input size="sm" aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('text-xs', 'py-1.5');
    });

    it('applies large size classes', () => {
      render(<Input size="lg" aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('text-base', 'py-2.5');
    });
  });

  describe('invalid state', () => {
    it('applies normal border by default', () => {
      render(<Input aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('border-slate-600');
      expect(input).not.toHaveClass('border-red-600');
    });

    it('applies error styling when invalid is true', () => {
      render(<Input invalid aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('border-red-600', 'focus:ring-red-500');
    });

    it('does not apply error styling when invalid is false', () => {
      render(<Input invalid={false} aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).not.toHaveClass('border-red-600');
    });
  });

  describe('base styling', () => {
    it('has full width', () => {
      render(<Input aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('w-full');
    });

    it('has rounded corners', () => {
      render(<Input aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('rounded-md');
    });

    it('has dark background', () => {
      render(<Input aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('bg-slate-900');
    });

    it('has proper focus ring styles', () => {
      render(<Input aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('focus:outline-none', 'focus:ring-2');
    });
  });

  describe('custom className', () => {
    it('applies custom className', () => {
      render(<Input className="custom-class" aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('custom-class');
    });

    it('merges custom className with default classes', () => {
      render(<Input className="custom-class" aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('custom-class', 'w-full', 'rounded-md');
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to input element', () => {
      const ref = React.createRef<HTMLInputElement>();
      render(<Input ref={ref} aria-label="test input" />);

      expect(ref.current).toBeInstanceOf(HTMLInputElement);
    });

    it('allows focus via ref', () => {
      const ref = React.createRef<HTMLInputElement>();
      render(<Input ref={ref} aria-label="test input" />);

      ref.current?.focus();
      expect(document.activeElement).toBe(ref.current);
    });
  });

  describe('HTML attributes', () => {
    it('passes through type attribute', () => {
      render(<Input type="password" aria-label="password input" />);
      const input = screen.getByLabelText('password input');
      expect(input).toHaveAttribute('type', 'password');
    });

    it('passes through placeholder attribute', () => {
      render(<Input placeholder="Enter text" aria-label="test input" />);
      expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
    });

    it('passes through disabled attribute', () => {
      render(<Input disabled aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toBeDisabled();
    });

    it('passes through required attribute', () => {
      render(<Input required aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toBeRequired();
    });

    it('passes through aria attributes', () => {
      render(<Input aria-describedby="helper-text" aria-label="test input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('aria-describedby', 'helper-text');
    });
  });

  describe('interactions', () => {
    it('calls onChange handler when typing', () => {
      const handleChange = jest.fn();
      render(<Input onChange={handleChange} aria-label="test input" />);

      const input = screen.getByRole('textbox');
      fireEvent.change(input, { target: { value: 'test' } });

      expect(handleChange).toHaveBeenCalledTimes(1);
    });

    it('calls onFocus handler when focused', () => {
      const handleFocus = jest.fn();
      render(<Input onFocus={handleFocus} aria-label="test input" />);

      const input = screen.getByRole('textbox');
      fireEvent.focus(input);

      expect(handleFocus).toHaveBeenCalledTimes(1);
    });

    it('calls onBlur handler when blurred', () => {
      const handleBlur = jest.fn();
      render(<Input onBlur={handleBlur} aria-label="test input" />);

      const input = screen.getByRole('textbox');
      fireEvent.blur(input);

      expect(handleBlur).toHaveBeenCalledTimes(1);
    });

    it('updates value when controlled', () => {
      const { rerender } = render(<Input value="" aria-label="test input" readOnly />);

      const input = screen.getByRole('textbox');
      expect(input).toHaveValue('');

      rerender(<Input value="new value" aria-label="test input" readOnly />);
      expect(input).toHaveValue('new value');
    });
  });
});
