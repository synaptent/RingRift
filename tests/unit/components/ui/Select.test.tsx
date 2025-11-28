import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Select, SelectSize } from '../../../../src/client/components/ui/Select';

describe('Select', () => {
  describe('rendering', () => {
    it('renders without crashing', () => {
      render(
        <Select aria-label="test select">
          <option value="1">Option 1</option>
          <option value="2">Option 2</option>
        </Select>
      );
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('renders children options correctly', () => {
      render(
        <Select aria-label="test select">
          <option value="a">Alpha</option>
          <option value="b">Beta</option>
          <option value="c">Gamma</option>
        </Select>
      );

      const select = screen.getByRole('combobox');
      expect(select).toContainHTML('<option value="a">Alpha</option>');
      expect(select).toContainHTML('<option value="b">Beta</option>');
      expect(select).toContainHTML('<option value="c">Gamma</option>');
    });

    it('renders with displayName set correctly', () => {
      expect(Select.displayName).toBe('Select');
    });
  });

  describe('sizes', () => {
    const sizes: SelectSize[] = ['sm', 'md', 'lg'];

    it.each(sizes)('renders %s size correctly', (size) => {
      render(
        <Select size={size} aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toBeInTheDocument();
    });

    it('applies medium size classes by default', () => {
      render(
        <Select aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('text-sm', 'py-2');
    });

    it('applies small size classes', () => {
      render(
        <Select size="sm" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('text-xs', 'py-1.5');
    });

    it('applies large size classes', () => {
      render(
        <Select size="lg" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('text-base', 'py-2.5');
    });
  });

  describe('invalid state', () => {
    it('applies normal border by default', () => {
      render(
        <Select aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('border-slate-600');
      expect(select).not.toHaveClass('border-red-600');
    });

    it('applies error styling when invalid is true', () => {
      render(
        <Select invalid aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('border-red-600', 'focus:ring-red-500');
    });

    it('does not apply error styling when invalid is false', () => {
      render(
        <Select invalid={false} aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).not.toHaveClass('border-red-600');
    });
  });

  describe('base styling', () => {
    it('has full width', () => {
      render(
        <Select aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('w-full');
    });

    it('has rounded corners', () => {
      render(
        <Select aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('rounded-lg');
    });

    it('has dark background', () => {
      render(
        <Select aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('bg-slate-900');
    });

    it('has proper focus ring styles', () => {
      render(
        <Select aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('focus:outline-none', 'focus:ring-2');
    });
  });

  describe('custom className', () => {
    it('applies custom className', () => {
      render(
        <Select className="custom-class" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('custom-class');
    });

    it('merges custom className with default classes', () => {
      render(
        <Select className="custom-class" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveClass('custom-class', 'w-full', 'rounded-lg');
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to select element', () => {
      const ref = React.createRef<HTMLSelectElement>();
      render(
        <Select ref={ref} aria-label="test select">
          <option>Test</option>
        </Select>
      );

      expect(ref.current).toBeInstanceOf(HTMLSelectElement);
    });

    it('allows focus via ref', () => {
      const ref = React.createRef<HTMLSelectElement>();
      render(
        <Select ref={ref} aria-label="test select">
          <option>Test</option>
        </Select>
      );

      ref.current?.focus();
      expect(document.activeElement).toBe(ref.current);
    });
  });

  describe('HTML attributes', () => {
    it('passes through disabled attribute', () => {
      render(
        <Select disabled aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toBeDisabled();
    });

    it('passes through required attribute', () => {
      render(
        <Select required aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toBeRequired();
    });

    it('passes through aria attributes', () => {
      render(
        <Select aria-describedby="helper-text" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('aria-describedby', 'helper-text');
    });

    it('passes through id attribute', () => {
      render(
        <Select id="my-select" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      expect(document.getElementById('my-select')).toBeInTheDocument();
    });

    it('passes through name attribute', () => {
      render(
        <Select name="selectField" aria-label="test select">
          <option>Test</option>
        </Select>
      );
      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('name', 'selectField');
    });
  });

  describe('interactions', () => {
    it('calls onChange handler when selection changes', () => {
      const handleChange = jest.fn();
      render(
        <Select onChange={handleChange} aria-label="test select">
          <option value="1">One</option>
          <option value="2">Two</option>
        </Select>
      );

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: '2' } });

      expect(handleChange).toHaveBeenCalledTimes(1);
    });

    it('calls onFocus handler when focused', () => {
      const handleFocus = jest.fn();
      render(
        <Select onFocus={handleFocus} aria-label="test select">
          <option>Test</option>
        </Select>
      );

      const select = screen.getByRole('combobox');
      fireEvent.focus(select);

      expect(handleFocus).toHaveBeenCalledTimes(1);
    });

    it('calls onBlur handler when blurred', () => {
      const handleBlur = jest.fn();
      render(
        <Select onBlur={handleBlur} aria-label="test select">
          <option>Test</option>
        </Select>
      );

      const select = screen.getByRole('combobox');
      fireEvent.blur(select);

      expect(handleBlur).toHaveBeenCalledTimes(1);
    });

    it('updates value when controlled', () => {
      const { rerender } = render(
        <Select value="1" aria-label="test select" onChange={() => {}}>
          <option value="1">One</option>
          <option value="2">Two</option>
        </Select>
      );

      const select = screen.getByRole('combobox');
      expect(select).toHaveValue('1');

      rerender(
        <Select value="2" aria-label="test select" onChange={() => {}}>
          <option value="1">One</option>
          <option value="2">Two</option>
        </Select>
      );
      expect(select).toHaveValue('2');
    });
  });
});
