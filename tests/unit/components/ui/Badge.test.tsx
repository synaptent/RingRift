import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Badge, BadgeVariant } from '../../../../src/client/components/ui/Badge';

describe('Badge', () => {
  describe('basic rendering', () => {
    it('renders without crashing', () => {
      render(<Badge>Test</Badge>);
      expect(screen.getByText('Test')).toBeInTheDocument();
    });

    it('renders children content', () => {
      render(<Badge>Badge Text</Badge>);
      expect(screen.getByText('Badge Text')).toBeInTheDocument();
    });

    it('renders as a span element', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge.tagName).toBe('SPAN');
    });

    it('renders with multiple children', () => {
      render(
        <Badge>
          <span>Icon</span>
          <span>Text</span>
        </Badge>
      );
      expect(screen.getByText('Icon')).toBeInTheDocument();
      expect(screen.getByText('Text')).toBeInTheDocument();
    });
  });

  describe('base styling', () => {
    it('applies base classes for inline-flex layout', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('inline-flex');
    });

    it('applies base classes for alignment', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('items-center');
    });

    it('applies base classes for rounded-full styling', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('rounded-full');
    });

    it('applies base classes for padding', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('px-2', 'py-0.5');
    });

    it('applies base classes for text size', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('text-xs');
    });

    it('applies base classes for font weight', () => {
      render(<Badge data-testid="badge">Test</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('font-semibold');
    });
  });

  describe('variant styling', () => {
    it('applies default variant styling when no variant specified', () => {
      render(<Badge data-testid="badge">Default</Badge>);
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('bg-slate-700', 'text-slate-100');
    });

    it('applies default variant styling explicitly', () => {
      render(
        <Badge data-testid="badge" variant="default">
          Default
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('bg-slate-700', 'text-slate-100');
    });

    it('applies primary variant styling', () => {
      render(
        <Badge data-testid="badge" variant="primary">
          Primary
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('bg-blue-600', 'text-white');
    });

    it('applies success variant styling', () => {
      render(
        <Badge data-testid="badge" variant="success">
          Success
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('bg-emerald-600', 'text-white');
    });

    it('applies warning variant styling', () => {
      render(
        <Badge data-testid="badge" variant="warning">
          Warning
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('bg-amber-500', 'text-slate-900');
    });

    it('applies danger variant styling', () => {
      render(
        <Badge data-testid="badge" variant="danger">
          Danger
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('bg-red-600', 'text-white');
    });

    it('applies outline variant styling', () => {
      render(
        <Badge data-testid="badge" variant="outline">
          Outline
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('border', 'border-slate-500', 'text-slate-200', 'bg-transparent');
    });
  });

  describe('custom className support', () => {
    it('accepts and applies custom className', () => {
      render(
        <Badge data-testid="badge" className="custom-class">
          Test
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('custom-class');
    });

    it('combines custom className with base and variant classes', () => {
      render(
        <Badge data-testid="badge" variant="primary" className="my-custom-class">
          Test
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('inline-flex', 'bg-blue-600', 'my-custom-class');
    });

    it('allows custom className to add extra padding', () => {
      render(
        <Badge data-testid="badge" className="px-4">
          Test
        </Badge>
      );
      const badge = screen.getByTestId('badge');
      expect(badge).toHaveClass('px-4');
    });
  });

  describe('HTML attribute passthrough', () => {
    it('passes through data-testid attribute', () => {
      render(<Badge data-testid="my-badge">Test</Badge>);
      expect(screen.getByTestId('my-badge')).toBeInTheDocument();
    });

    it('passes through id attribute', () => {
      render(<Badge id="badge-id">Test</Badge>);
      expect(document.getElementById('badge-id')).toBeInTheDocument();
    });

    it('passes through title attribute', () => {
      render(<Badge title="Badge tooltip">Test</Badge>);
      expect(screen.getByTitle('Badge tooltip')).toBeInTheDocument();
    });

    it('passes through aria-label attribute', () => {
      render(<Badge aria-label="Status badge">Test</Badge>);
      expect(screen.getByLabelText('Status badge')).toBeInTheDocument();
    });

    it('passes through onClick handler', () => {
      const handleClick = jest.fn();
      render(
        <Badge data-testid="badge" onClick={handleClick}>
          Clickable
        </Badge>
      );
      screen.getByTestId('badge').click();
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('passes through role attribute', () => {
      render(<Badge role="status">Active</Badge>);
      expect(screen.getByRole('status')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('renders text content for screen readers', () => {
      render(<Badge>Accessible Text</Badge>);
      expect(screen.getByText('Accessible Text')).toBeVisible();
    });

    it('supports aria-hidden for decorative badges', () => {
      render(<Badge aria-hidden="true">Decorative</Badge>);
      const badge = screen.getByText('Decorative');
      expect(badge).toHaveAttribute('aria-hidden', 'true');
    });
  });

  describe('variant type safety', () => {
    it('renders all variant types without error', () => {
      const variants: BadgeVariant[] = [
        'default',
        'primary',
        'success',
        'warning',
        'danger',
        'outline',
      ];

      variants.forEach((variant) => {
        const { unmount } = render(
          <Badge data-testid={`badge-${variant}`} variant={variant}>
            {variant}
          </Badge>
        );
        expect(screen.getByTestId(`badge-${variant}`)).toBeInTheDocument();
        unmount();
      });
    });
  });
});
