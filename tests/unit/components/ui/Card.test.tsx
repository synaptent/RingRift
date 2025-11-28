import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Card } from '../../../../src/client/components/ui/Card';

describe('Card', () => {
  describe('rendering', () => {
    it('renders without crashing', () => {
      render(<Card>Card content</Card>);
      expect(screen.getByText('Card content')).toBeInTheDocument();
    });

    it('renders children correctly', () => {
      render(
        <Card>
          <h2>Title</h2>
          <p>Description</p>
        </Card>
      );
      expect(screen.getByText('Title')).toBeInTheDocument();
      expect(screen.getByText('Description')).toBeInTheDocument();
    });

    it('renders as a div element', () => {
      const { container } = render(<Card>Content</Card>);
      const card = container.firstChild;
      expect(card?.nodeName).toBe('DIV');
    });
  });

  describe('padding', () => {
    it('applies padding by default (padded=true)', () => {
      const { container } = render(<Card>Padded Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('p-5');
    });

    it('applies padding when padded is explicitly true', () => {
      const { container } = render(<Card padded={true}>Padded Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('p-5');
    });

    it('does not apply padding when padded is false', () => {
      const { container } = render(<Card padded={false}>No Padding</Card>);
      const card = container.firstChild;
      expect(card).not.toHaveClass('p-5');
    });
  });

  describe('styling', () => {
    it('has rounded corners', () => {
      const { container } = render(<Card>Styled Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('rounded-2xl');
    });

    it('has border styling', () => {
      const { container } = render(<Card>Styled Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('border', 'border-slate-700');
    });

    it('has background styling', () => {
      const { container } = render(<Card>Styled Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('bg-slate-900/70');
    });

    it('has shadow styling', () => {
      const { container } = render(<Card>Styled Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('shadow-lg');
    });
  });

  describe('custom className', () => {
    it('applies custom className', () => {
      const { container } = render(<Card className="custom-class">Custom Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('custom-class');
    });

    it('merges custom className with default classes', () => {
      const { container } = render(<Card className="custom-class">Custom Card</Card>);
      const card = container.firstChild;
      expect(card).toHaveClass('custom-class', 'rounded-2xl', 'shadow-lg');
    });
  });

  describe('HTML attributes', () => {
    it('passes through id attribute', () => {
      render(<Card id="my-card">Card with ID</Card>);
      const card = document.getElementById('my-card');
      expect(card).toBeInTheDocument();
    });

    it('passes through data attributes', () => {
      const { container } = render(<Card data-testid="test-card">Test Card</Card>);
      expect(screen.getByTestId('test-card')).toBeInTheDocument();
    });

    it('passes through aria attributes', () => {
      const { container } = render(
        <Card role="region" aria-label="Card region">
          Accessible Card
        </Card>
      );
      const card = screen.getByRole('region');
      expect(card).toHaveAttribute('aria-label', 'Card region');
    });

    it('passes through onClick handler', () => {
      const handleClick = jest.fn();
      const { container } = render(<Card onClick={handleClick}>Clickable Card</Card>);
      const card = container.firstChild as HTMLElement;
      card.click();
      expect(handleClick).toHaveBeenCalledTimes(1);
    });
  });
});
