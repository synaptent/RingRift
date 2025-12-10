/**
 * Unit tests for QueueStatus component
 *
 * Tests the queue status display including:
 * - Visibility states (in queue, match found, neither)
 * - Queue information display (position, wait time)
 * - Cancel button functionality
 * - Match found display
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueueStatus } from '@/client/components/QueueStatus';
import type { MatchmakingPreferences } from '@/shared/types/websocket';

describe('QueueStatus', () => {
  const createPreferences = (overrides?: Partial<MatchmakingPreferences>): MatchmakingPreferences => ({
    boardType: 'square8',
    timeControl: { min: 300, max: 600 },
    ratingRange: { min: 1000, max: 1200 },
    ...overrides,
  });

  const defaultProps = {
    inQueue: false,
    estimatedWaitTime: null,
    queuePosition: null,
    searchCriteria: null,
    matchFound: false,
    onLeaveQueue: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('visibility', () => {
    it('should not render when not in queue and no match found', () => {
      const { container } = render(<QueueStatus {...defaultProps} />);
      expect(container.firstChild).toBeNull();
    });

    it('should render when in queue', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} />);
      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    it('should render when match found', () => {
      render(<QueueStatus {...defaultProps} matchFound={true} />);
      expect(screen.getByRole('status')).toBeInTheDocument();
    });
  });

  describe('queue display', () => {
    it('should show "Finding Opponent..." text when in queue', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} />);
      expect(screen.getByText('Finding Opponent...')).toBeInTheDocument();
    });

    it('should display queue position when provided', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} queuePosition={3} />);
      expect(screen.getByText('#3')).toBeInTheDocument();
    });

    it('should display estimated wait time in seconds when < 60s', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} estimatedWaitTime={45000} />);
      expect(screen.getByText('~45s')).toBeInTheDocument();
    });

    it('should display estimated wait time in minutes when >= 60s', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} estimatedWaitTime={90000} />);
      expect(screen.getByText('~1m 30s')).toBeInTheDocument();
    });

    it('should display just minutes when no remaining seconds', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} estimatedWaitTime={120000} />);
      expect(screen.getByText('~2m')).toBeInTheDocument();
    });

    it('should display "—" when queue position is null', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} queuePosition={null} />);
      const positionElements = screen.getAllByText('—');
      expect(positionElements.length).toBeGreaterThan(0);
    });

    it('should display search criteria when provided', () => {
      const preferences = createPreferences({ boardType: 'square8' });
      render(
        <QueueStatus {...defaultProps} inQueue={true} searchCriteria={preferences} />
      );
      expect(screen.getByText('Square 8×8')).toBeInTheDocument();
      expect(screen.getByText('1000–1200 Rating')).toBeInTheDocument();
    });
  });

  describe('cancel button', () => {
    it('should render cancel button when in queue', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} />);
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
    });

    it('should call onLeaveQueue when cancel button is clicked', () => {
      const onLeaveQueue = jest.fn();
      render(<QueueStatus {...defaultProps} inQueue={true} onLeaveQueue={onLeaveQueue} />);

      fireEvent.click(screen.getByRole('button', { name: /cancel/i }));

      expect(onLeaveQueue).toHaveBeenCalledTimes(1);
    });
  });

  describe('match found display', () => {
    it('should show "Match Found!" text when match is found', () => {
      render(<QueueStatus {...defaultProps} matchFound={true} />);
      expect(screen.getByText('Match Found!')).toBeInTheDocument();
    });

    it('should show "Joining game..." text when match is found', () => {
      render(<QueueStatus {...defaultProps} matchFound={true} />);
      expect(screen.getByText('Joining game...')).toBeInTheDocument();
    });

    it('should not show cancel button when match is found', () => {
      render(<QueueStatus {...defaultProps} matchFound={true} />);
      expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();
    });

    it('should have success styling when match is found', () => {
      render(<QueueStatus {...defaultProps} matchFound={true} />);
      const statusElement = screen.getByRole('status');
      expect(statusElement).toHaveClass('border-emerald-500');
    });
  });

  describe('board type formatting', () => {
    it('should format square8 as "Square 8×8"', () => {
      render(
        <QueueStatus
          {...defaultProps}
          inQueue={true}
          searchCriteria={createPreferences({ boardType: 'square8' })}
        />
      );
      expect(screen.getByText('Square 8×8')).toBeInTheDocument();
    });

    it('should format square19 as "Square 19×19"', () => {
      render(
        <QueueStatus
          {...defaultProps}
          inQueue={true}
          searchCriteria={createPreferences({ boardType: 'square19' })}
        />
      );
      expect(screen.getByText('Square 19×19')).toBeInTheDocument();
    });

    it('should format hexagonal as "Hexagonal"', () => {
      render(
        <QueueStatus
          {...defaultProps}
          inQueue={true}
          searchCriteria={createPreferences({ boardType: 'hexagonal' })}
        />
      );
      expect(screen.getByText('Hexagonal')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('should have role="status" for screen readers', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} />);
      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    it('should have aria-live="polite" for announcements', () => {
      render(<QueueStatus {...defaultProps} inQueue={true} />);
      expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
    });
  });
});
