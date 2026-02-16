import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MoveAnalysisPanel } from '../../../src/client/components/MoveAnalysisPanel';
import type { Move, Player } from '../../../src/shared/types/game';
import type { PositionEvaluationPayload } from '../../../src/shared/types/websocket';

describe('MoveAnalysisPanel', () => {
  const mockMove: Move = {
    type: 'move_stack',
    from: { x: 2, y: 3 },
    to: { x: 5, y: 3 },
  };

  const mockPlayers: Player[] = [
    {
      id: '1',
      username: 'Alice',
      playerNumber: 0,
      ringsInHand: 5,
      isAI: false,
      isEliminated: false,
    },
    { id: '2', username: 'Bob', playerNumber: 1, ringsInHand: 5, isAI: false, isEliminated: false },
  ];

  const mockEvaluation: PositionEvaluationPayload['data'] = {
    moveNumber: 5,
    perPlayer: {
      0: { totalEval: 2.5, territoryEval: 1.0, ringEval: 1.5 },
      1: { totalEval: -1.5, territoryEval: -0.5, ringEval: -1.0 },
    },
  };

  const mockPrevEvaluation: PositionEvaluationPayload['data'] = {
    moveNumber: 4,
    perPlayer: {
      0: { totalEval: 1.0, territoryEval: 0.5, ringEval: 0.5 },
      1: { totalEval: -0.5, territoryEval: -0.2, ringEval: -0.3 },
    },
  };

  const defaultProps = {
    analysis: {
      move: mockMove,
      moveNumber: 5,
      playerNumber: 0,
      evaluation: mockEvaluation,
      prevEvaluation: mockPrevEvaluation,
      thinkTimeMs: 1500,
      engineDepth: 12,
    },
    players: mockPlayers,
  };

  describe('empty state', () => {
    it('renders empty state when analysis is null', () => {
      render(<MoveAnalysisPanel analysis={null} players={mockPlayers} />);

      expect(screen.getByTestId('move-analysis-panel')).toBeInTheDocument();
      expect(screen.getByText('Move Analysis')).toBeInTheDocument();
      expect(screen.getByText('Select a move to see detailed analysis.')).toBeInTheDocument();
    });
  });

  describe('header display', () => {
    it('renders player name from players array', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      expect(screen.getByText('Alice')).toBeInTheDocument();
    });

    it('shows fallback player name when not found', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{ ...defaultProps.analysis!, playerNumber: 5 }}
          players={[]}
        />
      );

      expect(screen.getByText('Player 5')).toBeInTheDocument();
    });

    it('renders move number', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      expect(screen.getByText('Move #5')).toBeInTheDocument();
    });

    it('renders player color indicator', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      // Check for the color dot element
      const colorDot = screen.getByText('Alice').previousElementSibling;
      expect(colorDot).toHaveClass('rounded-full');
    });
  });

  describe('move type display', () => {
    const moveTypeTestCases = [
      { type: 'place_ring', expected: 'Placement' },
      { type: 'skip_placement', expected: 'Placement (pass)' },
      { type: 'no_placement_action', expected: 'Placement (pass)' },
      { type: 'move_stack', expected: 'Movement' },
      { type: 'move_ring', expected: 'Movement' },
      { type: 'overtaking_capture', expected: 'Capture' },
      { type: 'continue_capture_segment', expected: 'Capture' },
      { type: 'skip_capture', expected: 'End Capture' },
      { type: 'process_line', expected: 'Line' },
      { type: 'choose_line_option', expected: 'Line' },
      { type: 'choose_line_reward', expected: 'Line' },
      { type: 'no_line_action', expected: 'Line' },
      { type: 'process_territory_region', expected: 'Territory' },
      { type: 'choose_territory_option', expected: 'Territory' },
      { type: 'skip_territory_processing', expected: 'Territory' },
      { type: 'no_territory_action', expected: 'Territory' },
      { type: 'eliminate_rings_from_stack', expected: 'Remove Ring' },
      { type: 'forced_elimination', expected: 'Sacrifice' },
      { type: 'swap_sides', expected: 'Swap Sides' },
      { type: 'recovery_slide', expected: 'Recovery' },
      { type: 'skip_recovery', expected: 'Recovery' },
    ];

    moveTypeTestCases.forEach(({ type, expected }) => {
      it(`displays ${expected} for ${type} move type`, () => {
        render(
          <MoveAnalysisPanel
            {...defaultProps}
            analysis={{
              ...defaultProps.analysis!,
              move: { ...mockMove, type: type as Move['type'] },
            }}
          />
        );

        expect(screen.getByText(expected)).toBeInTheDocument();
      });
    });

    it('converts unknown move types by replacing underscores', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            move: { ...mockMove, type: 'custom_move_type' as Move['type'] },
          }}
        />
      );

      expect(screen.getByText('custom move type')).toBeInTheDocument();
    });
  });

  describe('quality badge', () => {
    it('shows Excellent badge for large positive delta', () => {
      const evaluation = {
        ...mockEvaluation,
        perPlayer: { 0: { totalEval: 5.0 }, 1: { totalEval: -2.0 } },
      };
      const prevEvaluation = {
        ...mockPrevEvaluation,
        perPlayer: { 0: { totalEval: 1.0 }, 1: { totalEval: 0.0 } },
      };

      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation,
            prevEvaluation,
          }}
        />
      );

      expect(screen.getByText('Excellent')).toBeInTheDocument();
    });

    it('shows Good badge for moderate positive delta', () => {
      const evaluation = {
        ...mockEvaluation,
        perPlayer: { 0: { totalEval: 2.5 }, 1: { totalEval: -1.0 } },
      };
      const prevEvaluation = {
        ...mockPrevEvaluation,
        perPlayer: { 0: { totalEval: 1.0 }, 1: { totalEval: 0.0 } },
      };

      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation,
            prevEvaluation,
          }}
        />
      );

      expect(screen.getByText('Good')).toBeInTheDocument();
    });

    it('shows Book badge for neutral delta', () => {
      const evaluation = {
        ...mockEvaluation,
        perPlayer: { 0: { totalEval: 1.0 }, 1: { totalEval: 0.0 } },
      };
      const prevEvaluation = {
        ...mockPrevEvaluation,
        perPlayer: { 0: { totalEval: 1.0 }, 1: { totalEval: 0.0 } },
      };

      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation,
            prevEvaluation,
          }}
        />
      );

      expect(screen.getByText('Book')).toBeInTheDocument();
    });

    it('shows Inaccuracy badge for small negative delta', () => {
      const evaluation = {
        ...mockEvaluation,
        perPlayer: { 0: { totalEval: -1.0 }, 1: { totalEval: 1.0 } },
      };
      const prevEvaluation = {
        ...mockPrevEvaluation,
        perPlayer: { 0: { totalEval: 1.0 }, 1: { totalEval: -1.0 } },
      };

      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation,
            prevEvaluation,
          }}
        />
      );

      expect(screen.getByText('Inaccuracy')).toBeInTheDocument();
    });

    it('shows Mistake badge for moderate negative delta', () => {
      const evaluation = {
        ...mockEvaluation,
        perPlayer: { 0: { totalEval: -4.0 }, 1: { totalEval: 3.0 } },
      };
      const prevEvaluation = {
        ...mockPrevEvaluation,
        perPlayer: { 0: { totalEval: 0.0 }, 1: { totalEval: 0.0 } },
      };

      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation,
            prevEvaluation,
          }}
        />
      );

      expect(screen.getByText('Mistake')).toBeInTheDocument();
    });

    it('shows Blunder badge for large negative delta', () => {
      const evaluation = {
        ...mockEvaluation,
        perPlayer: { 0: { totalEval: -8.0 }, 1: { totalEval: 6.0 } },
      };
      const prevEvaluation = {
        ...mockPrevEvaluation,
        perPlayer: { 0: { totalEval: 0.0 }, 1: { totalEval: 0.0 } },
      };

      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation,
            prevEvaluation,
          }}
        />
      );

      expect(screen.getByText('Blunder')).toBeInTheDocument();
    });

    it('does not show quality badge when no previous evaluation', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            prevEvaluation: undefined,
          }}
        />
      );

      expect(screen.queryByText('Excellent')).not.toBeInTheDocument();
      expect(screen.queryByText('Good')).not.toBeInTheDocument();
      expect(screen.queryByText('Blunder')).not.toBeInTheDocument();
    });
  });

  describe('evaluation details', () => {
    it('displays position eval with positive value', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      expect(screen.getByText('Position Eval:')).toBeInTheDocument();
      expect(screen.getByText('+2.5')).toBeInTheDocument();
    });

    it('displays eval change', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      expect(screen.getByText('Eval Change:')).toBeInTheDocument();
      expect(screen.getByText('+1.5')).toBeInTheDocument();
    });

    it('displays breakdown with territory and ring eval', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      expect(screen.getByText('Breakdown:')).toBeInTheDocument();
      expect(screen.getByText(/T:.*\+1\.0/)).toBeInTheDocument();
      expect(screen.getByText(/R:.*\+1\.5/)).toBeInTheDocument();
    });

    it('shows no evaluation message when evaluation is undefined', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            evaluation: undefined,
          }}
        />
      );

      expect(screen.getByText('No AI evaluation available for this move.')).toBeInTheDocument();
    });
  });

  describe('engine stats', () => {
    it('displays think time in milliseconds', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            thinkTimeMs: 500,
          }}
        />
      );

      expect(screen.getByText('Think Time:')).toBeInTheDocument();
      expect(screen.getByText('500ms')).toBeInTheDocument();
    });

    it('displays think time in seconds for longer times', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            thinkTimeMs: 5500,
          }}
        />
      );

      expect(screen.getByText('5.5s')).toBeInTheDocument();
    });

    it('displays think time in minutes for very long times', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            thinkTimeMs: 125000,
          }}
        />
      );

      expect(screen.getByText('2m 5s')).toBeInTheDocument();
    });

    it('displays search depth', () => {
      render(<MoveAnalysisPanel {...defaultProps} />);

      expect(screen.getByText('Search Depth:')).toBeInTheDocument();
      expect(screen.getByText('12')).toBeInTheDocument();
    });

    it('does not show engine stats when not provided', () => {
      render(
        <MoveAnalysisPanel
          {...defaultProps}
          analysis={{
            ...defaultProps.analysis!,
            thinkTimeMs: undefined,
            engineDepth: undefined,
          }}
        />
      );

      expect(screen.queryByText('Think Time:')).not.toBeInTheDocument();
      expect(screen.queryByText('Search Depth:')).not.toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      render(<MoveAnalysisPanel {...defaultProps} className="custom-class" />);

      expect(screen.getByTestId('move-analysis-panel')).toHaveClass('custom-class');
    });

    it('applies custom className to empty state', () => {
      render(<MoveAnalysisPanel analysis={null} players={mockPlayers} className="custom-class" />);

      expect(screen.getByTestId('move-analysis-panel')).toHaveClass('custom-class');
    });
  });
});
