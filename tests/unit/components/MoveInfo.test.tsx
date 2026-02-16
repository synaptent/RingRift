import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MoveInfo } from '../../../src/client/components/ReplayPanel/MoveInfo';
import type { ReplayMoveRecord } from '../../../src/client/types/replay';

describe('MoveInfo', () => {
  const mockMove: ReplayMoveRecord = {
    moveIndex: 5,
    player: 0,
    moveType: 'move_ring',
    move: {
      from: { x: 2, y: 3 },
      to: { x: 5, y: 3 },
    },
    thinkTimeMs: 1500,
    engineEval: 0.75,
    engineEvalType: 'cp',
    engineDepth: 12,
    enginePV: ['d4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6'],
    timeRemainingMs: 30000,
  };

  const defaultProps = {
    move: mockMove,
    moveNumber: 5,
  };

  describe('initial position', () => {
    it('shows initial position message when moveNumber is 0', () => {
      render(<MoveInfo move={null} moveNumber={0} />);

      expect(screen.getByText('Initial position (before any moves)')).toBeInTheDocument();
    });
  });

  describe('no move data', () => {
    it('shows no data message when move is null and moveNumber > 0', () => {
      render(<MoveInfo move={null} moveNumber={5} />);

      expect(screen.getByText('No move data available')).toBeInTheDocument();
    });
  });

  describe('move header', () => {
    it('displays move number', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('Move 5')).toBeInTheDocument();
    });

    it('displays player and move type', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('P0: Move (legacy)')).toBeInTheDocument();
    });

    it('displays think time when available', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('1500ms')).toBeInTheDocument();
    });

    it('does not display think time when null', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, thinkTimeMs: null }} />);

      expect(screen.queryByText(/ms$/)).not.toBeInTheDocument();
    });
  });

  describe('move type formatting', () => {
    const testCases = [
      { type: 'place_ring', expected: 'Place Ring' },
      { type: 'skip_placement', expected: 'Pass' },
      { type: 'move_ring', expected: 'Move (legacy)' },
      { type: 'move_stack', expected: 'Move' },
      { type: 'build_stack', expected: 'Build Stack (legacy)' },
      { type: 'overtaking_capture', expected: 'Capture' },
      { type: 'continue_capture_segment', expected: 'Continue Capture' },
      { type: 'process_line', expected: 'Score Line' },
      { type: 'choose_line_option', expected: 'Line Reward' },
      { type: 'choose_line_reward', expected: 'Line Reward (legacy)' },
      { type: 'process_territory_region', expected: 'Claim Territory (legacy)' },
      { type: 'choose_territory_option', expected: 'Claim Territory' },
      { type: 'eliminate_rings_from_stack', expected: 'Remove Ring' },
      { type: 'line_formation', expected: 'Line Scored (legacy)' },
      { type: 'territory_claim', expected: 'Territory Claimed (legacy)' },
      { type: 'chain_capture', expected: 'Chain Capture' },
      { type: 'forced_elimination', expected: 'Sacrifice' },
      { type: 'swap_sides', expected: 'Swap Sides' },
      { type: 'recovery_slide', expected: 'Recovery' },
      { type: 'skip_recovery', expected: 'Skip Recovery' },
    ];

    testCases.forEach(({ type, expected }) => {
      it(`formats ${type} as ${expected}`, () => {
        render(<MoveInfo {...defaultProps} move={{ ...mockMove, moveType: type }} />);

        expect(screen.getByText(`P0: ${expected}`)).toBeInTheDocument();
      });
    });

    it('converts unknown types by replacing underscores with spaces', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, moveType: 'custom_move_type' }} />);

      expect(screen.getByText('P0: custom move type')).toBeInTheDocument();
    });
  });

  describe('position display', () => {
    it('displays from position', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('(2,3)')).toBeInTheDocument();
    });

    it('displays to position', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('(5,3)')).toBeInTheDocument();
    });

    it('displays arrow between positions', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('→')).toBeInTheDocument();
    });

    it('displays 3D position with z coordinate', () => {
      render(
        <MoveInfo
          {...defaultProps}
          move={{
            ...mockMove,
            move: {
              from: { x: 1, y: 2, z: 3 },
              to: { x: 4, y: 5, z: 6 },
            },
          }}
        />
      );

      expect(screen.getByText('(1,2,3)')).toBeInTheDocument();
      expect(screen.getByText('(4,5,6)')).toBeInTheDocument();
    });

    it('does not display positions section when neither from nor to exist', () => {
      render(
        <MoveInfo
          {...defaultProps}
          move={{
            ...mockMove,
            move: {},
          }}
        />
      );

      expect(screen.queryByText('→')).not.toBeInTheDocument();
    });

    it('displays only from position when to is missing', () => {
      render(
        <MoveInfo
          {...defaultProps}
          move={{
            ...mockMove,
            move: { from: { x: 2, y: 3 } },
          }}
        />
      );

      expect(screen.getByText('(2,3)')).toBeInTheDocument();
      expect(screen.queryByText('→')).not.toBeInTheDocument();
    });

    it('displays only to position when from is missing', () => {
      render(
        <MoveInfo
          {...defaultProps}
          move={{
            ...mockMove,
            move: { to: { x: 5, y: 3 } },
          }}
        />
      );

      expect(screen.getByText('(5,3)')).toBeInTheDocument();
      expect(screen.queryByText('→')).not.toBeInTheDocument();
    });
  });

  describe('engine evaluation', () => {
    it('displays positive eval with plus sign', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: 1.5 }} />);

      expect(screen.getByText('+1.50')).toBeInTheDocument();
    });

    it('displays negative eval', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: -0.75 }} />);

      expect(screen.getByText('-0.75')).toBeInTheDocument();
    });

    it('displays zero eval with plus sign', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: 0 }} />);

      expect(screen.getByText('+0.00')).toBeInTheDocument();
    });

    it('displays eval type', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('(cp)')).toBeInTheDocument();
    });

    it('displays engine depth', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('d12')).toBeInTheDocument();
    });

    it('does not display eval section when engineEval is undefined', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: undefined }} />);

      expect(screen.queryByText('Eval:')).not.toBeInTheDocument();
    });

    it('does not display eval section when engineEval is null', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: null }} />);

      expect(screen.queryByText('Eval:')).not.toBeInTheDocument();
    });

    it('applies green color to positive eval', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: 1.0 }} />);

      const evalElement = screen.getByText('+1.00');
      expect(evalElement).toHaveClass('text-emerald-400');
    });

    it('applies red color to negative eval', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: -1.0 }} />);

      const evalElement = screen.getByText('-1.00');
      expect(evalElement).toHaveClass('text-red-400');
    });

    it('applies neutral color to zero eval', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, engineEval: 0 }} />);

      const evalElement = screen.getByText('+0.00');
      expect(evalElement).toHaveClass('text-slate-300');
    });
  });

  describe('principal variation', () => {
    it('displays PV label', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('PV:')).toBeInTheDocument();
    });

    it('displays first 5 moves of PV', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('d4 e5 Nf3 Nc6 Bb5')).toBeInTheDocument();
    });

    it('shows ellipsis when PV has more than 5 moves', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('...')).toBeInTheDocument();
    });

    it('does not show ellipsis when PV has 5 or fewer moves', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, enginePV: ['d4', 'e5', 'Nf3'] }} />);

      expect(screen.queryByText('...')).not.toBeInTheDocument();
    });

    it('does not display PV section when enginePV is empty', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, enginePV: [] }} />);

      expect(screen.queryByText('PV:')).not.toBeInTheDocument();
    });

    it('does not display PV section when enginePV is undefined', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, enginePV: undefined }} />);

      expect(screen.queryByText('PV:')).not.toBeInTheDocument();
    });
  });

  describe('time remaining', () => {
    it('displays time remaining in seconds', () => {
      render(<MoveInfo {...defaultProps} />);

      expect(screen.getByText('Clock: 30s remaining')).toBeInTheDocument();
    });

    it('converts milliseconds to seconds correctly', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, timeRemainingMs: 90000 }} />);

      expect(screen.getByText('Clock: 90s remaining')).toBeInTheDocument();
    });

    it('does not display time section when timeRemainingMs is undefined', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, timeRemainingMs: undefined }} />);

      expect(screen.queryByText(/Clock:/)).not.toBeInTheDocument();
    });

    it('does not display time section when timeRemainingMs is null', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, timeRemainingMs: null }} />);

      expect(screen.queryByText(/Clock:/)).not.toBeInTheDocument();
    });
  });

  describe('player display', () => {
    it('displays player 0', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, player: 0 }} />);

      expect(screen.getByText(/P0:/)).toBeInTheDocument();
    });

    it('displays player 1', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, player: 1 }} />);

      expect(screen.getByText(/P1:/)).toBeInTheDocument();
    });

    it('displays player 3', () => {
      render(<MoveInfo {...defaultProps} move={{ ...mockMove, player: 3 }} />);

      expect(screen.getByText(/P3:/)).toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      const { container } = render(<MoveInfo {...defaultProps} className="custom-class" />);

      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('applies custom className to initial position message', () => {
      const { container } = render(
        <MoveInfo move={null} moveNumber={0} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('applies custom className to no data message', () => {
      const { container } = render(
        <MoveInfo move={null} moveNumber={5} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
