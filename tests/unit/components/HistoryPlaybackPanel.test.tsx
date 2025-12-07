import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { HistoryPlaybackPanel } from '../../../src/client/components/HistoryPlaybackPanel';

describe('HistoryPlaybackPanel', () => {
  const defaultProps = {
    totalMoves: 10,
    currentMoveIndex: 5,
    isViewingHistory: true,
    onMoveIndexChange: jest.fn(),
    onExitHistoryView: jest.fn(),
    onEnterHistoryView: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders when visible and has moves', () => {
    render(<HistoryPlaybackPanel {...defaultProps} />);

    expect(screen.getByText('History Playback')).toBeInTheDocument();
    expect(screen.getByLabelText('Move scrubber')).toBeInTheDocument();
  });

  it('does not render when visible is false', () => {
    render(<HistoryPlaybackPanel {...defaultProps} visible={false} />);

    expect(screen.queryByText('History Playback')).not.toBeInTheDocument();
  });

  it('does not render when totalMoves is 0', () => {
    render(<HistoryPlaybackPanel {...defaultProps} totalMoves={0} />);

    expect(screen.queryByText('History Playback')).not.toBeInTheDocument();
  });

  it('shows Return to Live button when viewing history', () => {
    render(<HistoryPlaybackPanel {...defaultProps} />);

    expect(screen.getByText('Return to Live')).toBeInTheDocument();
  });

  it('hides Return to Live button when not viewing history', () => {
    render(<HistoryPlaybackPanel {...defaultProps} isViewingHistory={false} />);

    expect(screen.queryByText('Return to Live')).not.toBeInTheDocument();
  });

  it('calls onExitHistoryView when Return to Live is clicked', () => {
    render(<HistoryPlaybackPanel {...defaultProps} />);

    fireEvent.click(screen.getByText('Return to Live'));
    expect(defaultProps.onExitHistoryView).toHaveBeenCalledTimes(1);
  });

  it('displays current move position in scrubber', () => {
    render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={3} totalMoves={10} />);

    expect(screen.getByText('Move 3 / 10')).toBeInTheDocument();
  });

  it('shows total moves when not viewing history', () => {
    render(
      <HistoryPlaybackPanel
        {...defaultProps}
        isViewingHistory={false}
        currentMoveIndex={3}
        totalMoves={10}
      />
    );

    expect(screen.getByText('Move 10 / 10')).toBeInTheDocument();
  });

  describe('playback controls', () => {
    it('step back button calls onMoveIndexChange with decremented index', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} />);

      fireEvent.click(screen.getByLabelText('Step back'));
      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(4);
    });

    it('step forward button calls onMoveIndexChange with incremented index', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} totalMoves={10} />);

      fireEvent.click(screen.getByLabelText('Step forward'));
      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(6);
    });

    it('jump to start button moves to index 0', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} />);

      fireEvent.click(screen.getByLabelText('Jump to start'));
      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(0);
    });

    it('jump to end button moves to totalMoves', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} totalMoves={10} />);

      fireEvent.click(screen.getByLabelText('Jump to end'));
      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(10);
    });

    it('step back is disabled at index 0', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={0} />);

      expect(screen.getByLabelText('Step back')).toBeDisabled();
    });

    it('step forward is disabled at totalMoves', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={10} totalMoves={10} />);

      expect(screen.getByLabelText('Step forward')).toBeDisabled();
    });

    it('entering history mode when stepping from non-history view', () => {
      render(
        <HistoryPlaybackPanel {...defaultProps} isViewingHistory={false} currentMoveIndex={5} />
      );

      fireEvent.click(screen.getByLabelText('Step back'));

      expect(defaultProps.onEnterHistoryView).toHaveBeenCalled();
      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(4);
    });
  });

  describe('scrubber', () => {
    it('calls onMoveIndexChange when scrubber value changes', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const scrubber = screen.getByLabelText('Move scrubber');
      fireEvent.change(scrubber, { target: { value: '7' } });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(7);
    });

    it('enters history view when scrubbing from non-history state', () => {
      render(<HistoryPlaybackPanel {...defaultProps} isViewingHistory={false} />);

      const scrubber = screen.getByLabelText('Move scrubber');
      fireEvent.change(scrubber, { target: { value: '3' } });

      expect(defaultProps.onEnterHistoryView).toHaveBeenCalled();
    });
  });

  describe('speed controls', () => {
    it('renders all speed options', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByText('0.5x')).toBeInTheDocument();
      expect(screen.getByText('1x')).toBeInTheDocument();
      expect(screen.getByText('2x')).toBeInTheDocument();
      expect(screen.getByText('5x')).toBeInTheDocument();
    });

    it('highlights the selected speed', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      // 1x is default selected
      const oneXButton = screen.getByText('1x');
      expect(oneXButton).toHaveClass('bg-emerald-900/40');
    });

    it('changes speed when a different speed is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('2x'));

      expect(screen.getByText('2x')).toHaveClass('bg-emerald-900/40');
      expect(screen.getByText('1x')).not.toHaveClass('bg-emerald-900/40');
    });
  });

  describe('auto-play', () => {
    it('starts playback when play button is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={0} />);

      fireEvent.click(screen.getByLabelText('Play'));

      // After clicking play, button should show pause
      expect(screen.getByLabelText('Pause')).toBeInTheDocument();
    });

    it('pauses playback when pause button is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={0} />);

      fireEvent.click(screen.getByLabelText('Play'));
      fireEvent.click(screen.getByLabelText('Pause'));

      expect(screen.getByLabelText('Play')).toBeInTheDocument();
    });

    it('advances moves at playback speed during autoplay', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={0} />);

      fireEvent.click(screen.getByLabelText('Play'));

      // At 1x speed, should advance every 1000ms
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(1);
    });

    it('respects faster playback speed', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={0} />);

      // Set 2x speed
      fireEvent.click(screen.getByText('2x'));
      fireEvent.click(screen.getByLabelText('Play'));

      // At 2x speed, should advance every 500ms
      act(() => {
        jest.advanceTimersByTime(500);
      });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(1);
    });

    it('enters history view and starts from beginning when playing from non-history mode', () => {
      render(
        <HistoryPlaybackPanel {...defaultProps} isViewingHistory={false} currentMoveIndex={5} />
      );

      fireEvent.click(screen.getByLabelText('Play'));

      expect(defaultProps.onEnterHistoryView).toHaveBeenCalled();
      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(0);
    });
  });

  describe('keyboard shortcuts', () => {
    it('ArrowLeft triggers step back', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} />);

      fireEvent.keyDown(window, { key: 'ArrowLeft' });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(4);
    });

    it('ArrowRight triggers step forward', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} totalMoves={10} />);

      fireEvent.keyDown(window, { key: 'ArrowRight' });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(6);
    });

    it('Space toggles play/pause', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={0} />);

      fireEvent.keyDown(window, { key: ' ' });

      expect(screen.getByLabelText('Pause')).toBeInTheDocument();
    });

    it('Home jumps to start', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} />);

      fireEvent.keyDown(window, { key: 'Home' });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(0);
    });

    it('End jumps to end', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} totalMoves={10} />);

      fireEvent.keyDown(window, { key: 'End' });

      expect(defaultProps.onMoveIndexChange).toHaveBeenCalledWith(10);
    });

    it('Escape exits history view', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.keyDown(window, { key: 'Escape' });

      expect(defaultProps.onExitHistoryView).toHaveBeenCalled();
    });

    it('ignores keyboard shortcuts when target is input', () => {
      render(<HistoryPlaybackPanel {...defaultProps} currentMoveIndex={5} />);

      // Create an input element and dispatch keydown from it
      const inputElement = document.createElement('input');
      document.body.appendChild(inputElement);

      fireEvent.keyDown(inputElement, { key: 'ArrowLeft' });

      expect(defaultProps.onMoveIndexChange).not.toHaveBeenCalled();

      document.body.removeChild(inputElement);
    });

    it('does not respond to shortcuts when not visible', () => {
      render(<HistoryPlaybackPanel {...defaultProps} visible={false} currentMoveIndex={5} />);

      fireEvent.keyDown(window, { key: 'ArrowLeft' });

      expect(defaultProps.onMoveIndexChange).not.toHaveBeenCalled();
    });
  });

  describe('hasSnapshots=false', () => {
    it('shows unavailable message when hasSnapshots is false', () => {
      render(<HistoryPlaybackPanel {...defaultProps} hasSnapshots={false} />);

      expect(screen.getByText(/History scrubbing is unavailable/)).toBeInTheDocument();
    });

    it('disables controls when hasSnapshots is false', () => {
      render(<HistoryPlaybackPanel {...defaultProps} hasSnapshots={false} />);

      expect(screen.getByLabelText('Step back')).toBeDisabled();
      expect(screen.getByLabelText('Step forward')).toBeDisabled();
      expect(screen.getByLabelText('Jump to start')).toBeDisabled();
      expect(screen.getByLabelText('Jump to end')).toBeDisabled();
      expect(screen.getByLabelText('Move scrubber')).toBeDisabled();
    });

    it('hides Return to Live button when hasSnapshots is false', () => {
      render(<HistoryPlaybackPanel {...defaultProps} hasSnapshots={false} />);

      expect(screen.queryByText('Return to Live')).not.toBeInTheDocument();
    });
  });

  describe('keyboard hints', () => {
    it('displays keyboard shortcut hints', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByText('Step')).toBeInTheDocument();
      expect(screen.getByText('Play/Pause')).toBeInTheDocument();
      expect(screen.getByText('Exit')).toBeInTheDocument();
    });
  });
});
