import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { PlaybackControls } from '../../../src/client/components/ReplayPanel/PlaybackControls';

describe('PlaybackControls', () => {
  const defaultProps = {
    currentMove: 5,
    totalMoves: 20,
    isPlaying: false,
    playbackSpeed: 1 as const,
    canStepForward: true,
    canStepBackward: true,
    onStepForward: jest.fn(),
    onStepBackward: jest.fn(),
    onJumpToStart: jest.fn(),
    onJumpToEnd: jest.fn(),
    onJumpToMove: jest.fn(),
    onTogglePlay: jest.fn(),
    onSetSpeed: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('transport buttons rendering', () => {
    it('renders jump to start button', () => {
      render(<PlaybackControls {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Jump to start/i })).toBeInTheDocument();
    });

    it('renders step backward button', () => {
      render(<PlaybackControls {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Step backward/i })).toBeInTheDocument();
    });

    it('renders play button when not playing', () => {
      render(<PlaybackControls {...defaultProps} isPlaying={false} />);

      expect(screen.getByRole('button', { name: /Play/i })).toBeInTheDocument();
    });

    it('renders pause button when playing', () => {
      render(<PlaybackControls {...defaultProps} isPlaying={true} />);

      expect(screen.getByRole('button', { name: /Pause/i })).toBeInTheDocument();
    });

    it('renders step forward button', () => {
      render(<PlaybackControls {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Step forward/i })).toBeInTheDocument();
    });

    it('renders jump to end button', () => {
      render(<PlaybackControls {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Jump to end/i })).toBeInTheDocument();
    });
  });

  describe('transport button interactions', () => {
    it('calls onJumpToStart when jump to start button is clicked', async () => {
      const onJumpToStart = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onJumpToStart={onJumpToStart} />);

      await user.click(screen.getByRole('button', { name: /Jump to start/i }));

      expect(onJumpToStart).toHaveBeenCalledTimes(1);
    });

    it('calls onStepBackward when step backward button is clicked', async () => {
      const onStepBackward = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onStepBackward={onStepBackward} />);

      await user.click(screen.getByRole('button', { name: /Step backward/i }));

      expect(onStepBackward).toHaveBeenCalledTimes(1);
    });

    it('calls onTogglePlay when play button is clicked', async () => {
      const onTogglePlay = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onTogglePlay={onTogglePlay} />);

      await user.click(screen.getByRole('button', { name: /Play/i }));

      expect(onTogglePlay).toHaveBeenCalledTimes(1);
    });

    it('calls onTogglePlay when pause button is clicked', async () => {
      const onTogglePlay = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} isPlaying={true} onTogglePlay={onTogglePlay} />);

      await user.click(screen.getByRole('button', { name: /Pause/i }));

      expect(onTogglePlay).toHaveBeenCalledTimes(1);
    });

    it('calls onStepForward when step forward button is clicked', async () => {
      const onStepForward = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onStepForward={onStepForward} />);

      await user.click(screen.getByRole('button', { name: /Step forward/i }));

      expect(onStepForward).toHaveBeenCalledTimes(1);
    });

    it('calls onJumpToEnd when jump to end button is clicked', async () => {
      const onJumpToEnd = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onJumpToEnd={onJumpToEnd} />);

      await user.click(screen.getByRole('button', { name: /Jump to end/i }));

      expect(onJumpToEnd).toHaveBeenCalledTimes(1);
    });
  });

  describe('disabled states', () => {
    it('disables backward buttons when canStepBackward is false', () => {
      render(<PlaybackControls {...defaultProps} canStepBackward={false} />);

      expect(screen.getByRole('button', { name: /Jump to start/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /Step backward/i })).toBeDisabled();
    });

    it('disables forward buttons when canStepForward is false', () => {
      render(<PlaybackControls {...defaultProps} canStepForward={false} />);

      expect(screen.getByRole('button', { name: /Step forward/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /Jump to end/i })).toBeDisabled();
    });

    it('disables play button when canStepForward is false and not playing', () => {
      render(<PlaybackControls {...defaultProps} canStepForward={false} isPlaying={false} />);

      expect(screen.getByRole('button', { name: /Play/i })).toBeDisabled();
    });

    it('does not disable pause button when playing even if canStepForward is false', () => {
      render(<PlaybackControls {...defaultProps} canStepForward={false} isPlaying={true} />);

      expect(screen.getByRole('button', { name: /Pause/i })).not.toBeDisabled();
    });

    it('disables all transport buttons when isLoading is true', () => {
      render(<PlaybackControls {...defaultProps} isLoading={true} />);

      expect(screen.getByRole('button', { name: /Jump to start/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /Step backward/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /Step forward/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /Jump to end/i })).toBeDisabled();
    });
  });

  describe('speed selector', () => {
    it('renders all speed options', () => {
      render(<PlaybackControls {...defaultProps} />);

      expect(screen.getByRole('button', { name: /0\.5x/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /1x/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /2x/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /4x/i })).toBeInTheDocument();
    });

    it('shows Speed label', () => {
      render(<PlaybackControls {...defaultProps} />);

      expect(screen.getByText('Speed:')).toBeInTheDocument();
    });

    it('calls onSetSpeed when speed button is clicked', async () => {
      const onSetSpeed = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onSetSpeed={onSetSpeed} />);

      await user.click(screen.getByRole('button', { name: /2x/i }));

      expect(onSetSpeed).toHaveBeenCalledWith(2);
    });

    it('calls onSetSpeed with 0.5 when slow speed is clicked', async () => {
      const onSetSpeed = jest.fn();
      const user = userEvent.setup();
      render(<PlaybackControls {...defaultProps} onSetSpeed={onSetSpeed} />);

      await user.click(screen.getByRole('button', { name: /0\.5x/i }));

      expect(onSetSpeed).toHaveBeenCalledWith(0.5);
    });

    it('highlights currently selected speed', () => {
      render(<PlaybackControls {...defaultProps} playbackSpeed={2} />);

      const speed2Button = screen.getByRole('button', { name: /2x/i });
      expect(speed2Button).toHaveClass('border-emerald-500/50');
      expect(speed2Button).toHaveClass('bg-emerald-900/40');
    });
  });

  describe('progress bar / scrubber', () => {
    it('renders visual slider with correct aria attributes', () => {
      render(<PlaybackControls {...defaultProps} currentMove={5} totalMoves={20} />);

      // Get the visible slider (div with role="slider"), not the hidden input
      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const visualSlider = sliders.find((el) => el.tagName === 'DIV');

      expect(visualSlider).toHaveAttribute('aria-valuemin', '0');
      expect(visualSlider).toHaveAttribute('aria-valuemax', '20');
      expect(visualSlider).toHaveAttribute('aria-valuenow', '5');
    });

    it('calls onJumpToMove when scrubber is clicked', () => {
      const onJumpToMove = jest.fn();
      render(<PlaybackControls {...defaultProps} onJumpToMove={onJumpToMove} totalMoves={100} />);

      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const visualSlider = sliders.find((el) => el.tagName === 'DIV')!;

      // Mock getBoundingClientRect
      jest.spyOn(visualSlider, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        right: 100,
        width: 100,
        top: 0,
        bottom: 10,
        height: 10,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      // Click at 50% of the scrubber
      fireEvent.click(visualSlider, { clientX: 50 });

      expect(onJumpToMove).toHaveBeenCalledWith(50);
    });

    it('clamps scrubber click to valid range', () => {
      const onJumpToMove = jest.fn();
      render(<PlaybackControls {...defaultProps} onJumpToMove={onJumpToMove} totalMoves={20} />);

      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const visualSlider = sliders.find((el) => el.tagName === 'DIV')!;

      jest.spyOn(visualSlider, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        right: 100,
        width: 100,
        top: 0,
        bottom: 10,
        height: 10,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      // Click beyond 100%
      fireEvent.click(visualSlider, { clientX: 150 });

      // Should clamp to totalMoves (20)
      expect(onJumpToMove).toHaveBeenCalledWith(20);
    });
  });

  describe('hidden range input', () => {
    it('renders hidden range input for keyboard accessibility', () => {
      render(<PlaybackControls {...defaultProps} currentMove={5} totalMoves={20} />);

      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const rangeInput = sliders.find((el) => el.tagName === 'INPUT');

      expect(rangeInput).toHaveAttribute('type', 'range');
      expect(rangeInput).toHaveAttribute('min', '0');
      expect(rangeInput).toHaveAttribute('max', '20');
      expect(rangeInput).toHaveValue('5');
    });

    it('calls onJumpToMove when range input changes', () => {
      const onJumpToMove = jest.fn();
      render(<PlaybackControls {...defaultProps} onJumpToMove={onJumpToMove} />);

      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const hiddenInput = sliders.find((el) => el.tagName === 'INPUT');

      if (hiddenInput) {
        fireEvent.change(hiddenInput, { target: { value: '15' } });
        expect(onJumpToMove).toHaveBeenCalledWith(15);
      }
    });
  });

  describe('move counter', () => {
    it('displays current move', () => {
      render(<PlaybackControls {...defaultProps} currentMove={7} />);

      expect(screen.getByText('Move 7')).toBeInTheDocument();
    });

    it('displays total moves', () => {
      render(<PlaybackControls {...defaultProps} totalMoves={25} />);

      expect(screen.getByText('of 25')).toBeInTheDocument();
    });

    it('updates when current move changes', () => {
      const { rerender } = render(<PlaybackControls {...defaultProps} currentMove={3} />);

      expect(screen.getByText('Move 3')).toBeInTheDocument();

      rerender(<PlaybackControls {...defaultProps} currentMove={10} />);

      expect(screen.getByText('Move 10')).toBeInTheDocument();
    });
  });

  describe('progress calculation', () => {
    it('shows 0% progress when currentMove is 0', () => {
      render(<PlaybackControls {...defaultProps} currentMove={0} totalMoves={20} />);

      // Get the visual slider (div with role="slider")
      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const visualSlider = sliders.find((el) => el.tagName === 'DIV');
      expect(visualSlider).toHaveAttribute('aria-valuenow', '0');
    });

    it('shows 100% progress when currentMove equals totalMoves', () => {
      render(<PlaybackControls {...defaultProps} currentMove={20} totalMoves={20} />);

      const sliders = screen.getAllByRole('slider', { name: /Playback position/i });
      const visualSlider = sliders.find((el) => el.tagName === 'DIV');
      expect(visualSlider).toHaveAttribute('aria-valuenow', '20');
    });

    it('handles totalMoves of 0 without error', () => {
      // Should not throw an error (division by zero)
      render(<PlaybackControls {...defaultProps} currentMove={0} totalMoves={0} />);

      expect(screen.getByText('Move 0')).toBeInTheDocument();
      expect(screen.getByText('of 0')).toBeInTheDocument();
    });
  });

  describe('play button styling', () => {
    it('applies playing state styling when isPlaying is true', () => {
      render(<PlaybackControls {...defaultProps} isPlaying={true} />);

      const pauseButton = screen.getByRole('button', { name: /Pause/i });
      expect(pauseButton).toHaveClass('bg-emerald-900/40');
      expect(pauseButton).toHaveClass('border-emerald-500/50');
    });

    it('does not apply playing state styling when isPlaying is false', () => {
      render(<PlaybackControls {...defaultProps} isPlaying={false} />);

      const playButton = screen.getByRole('button', { name: /Play/i });
      expect(playButton).not.toHaveClass('bg-emerald-900/40');
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      const { container } = render(<PlaybackControls {...defaultProps} className="custom-class" />);

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
