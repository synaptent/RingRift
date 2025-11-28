import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ErrorBoundary from '../../../src/client/components/ErrorBoundary';

// Mock the reportClientError function
jest.mock('../../../src/client/utils/errorReporting', () => ({
  reportClientError: jest.fn(),
}));

import { reportClientError } from '../../../src/client/utils/errorReporting';

// Component that throws an error
const ErrorThrowingComponent: React.FC<{ shouldThrow?: boolean }> = ({ shouldThrow = true }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};

// Suppress console.error output during error boundary tests
const originalError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalError;
});

describe('ErrorBoundary', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('normal rendering', () => {
    it('renders children when there is no error', () => {
      render(
        <ErrorBoundary>
          <div>Child content</div>
        </ErrorBoundary>
      );
      expect(screen.getByText('Child content')).toBeInTheDocument();
    });

    it('renders multiple children correctly', () => {
      render(
        <ErrorBoundary>
          <div>First child</div>
          <div>Second child</div>
        </ErrorBoundary>
      );
      expect(screen.getByText('First child')).toBeInTheDocument();
      expect(screen.getByText('Second child')).toBeInTheDocument();
    });
  });

  describe('error handling', () => {
    it('renders fallback UI when a child throws an error', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('shows helpful message in fallback UI', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByText(/An unexpected error occurred/)).toBeInTheDocument();
      expect(screen.getByText(/Try refreshing the page/)).toBeInTheDocument();
    });

    it('renders reload button in fallback UI', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      const reloadButton = screen.getByRole('button', { name: /Reload page/i });
      expect(reloadButton).toBeInTheDocument();
    });

    it('calls reportClientError when error is caught', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      expect(reportClientError).toHaveBeenCalled();
      expect(reportClientError).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({
          type: 'react_render_error',
        })
      );
    });

    it('passes componentStack to reportClientError', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      expect(reportClientError).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({
          componentStack: expect.any(String),
        })
      );
    });
  });

  describe('reload functionality', () => {
    it('calls window.location.reload when reload button is clicked', () => {
      // Mock window.location.reload
      const reloadMock = jest.fn();
      Object.defineProperty(window, 'location', {
        value: { reload: reloadMock },
        writable: true,
      });

      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      const reloadButton = screen.getByRole('button', { name: /Reload page/i });
      fireEvent.click(reloadButton);

      expect(reloadMock).toHaveBeenCalledTimes(1);
    });
  });

  describe('fallback UI styling', () => {
    it('has proper layout classes for centering', () => {
      const { container } = render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      const fallbackContainer = container.firstChild as HTMLElement;
      expect(fallbackContainer).toHaveClass(
        'min-h-screen',
        'flex',
        'flex-col',
        'items-center',
        'justify-center'
      );
    });

    it('has proper background styling', () => {
      const { container } = render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      const fallbackContainer = container.firstChild as HTMLElement;
      expect(fallbackContainer).toHaveClass('bg-slate-950');
    });
  });

  describe('accessibility', () => {
    it('has properly labeled reload button', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('type', 'button');
    });

    it('has heading structure', () => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );

      const heading = screen.getByRole('heading', { level: 1 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent('Something went wrong');
    });
  });
});
