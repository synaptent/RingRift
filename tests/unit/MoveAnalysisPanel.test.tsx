import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import {
  MoveAnalysisPanel,
  type MoveAnalysis,
} from '../../src/client/components/MoveAnalysisPanel';
import type { PositionEvaluationPayload } from '../../src/shared/types/websocket';
import type { Player, Move, GamePhase } from '../../src/shared/types/game';

// Helper to create test players
function createTestPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

// Map phases to appropriate move types for testing getMoveTypeLabel
function getMoveTypeForPhase(phase: GamePhase): string {
  switch (phase) {
    case 'ring_placement':
      return 'place_ring';
    case 'movement':
      return 'move_stack';
    case 'capture':
      return 'overtaking_capture';
    case 'chain_capture':
      return 'continue_capture_segment';
    case 'line_processing':
      return 'process_line';
    case 'territory_processing':
      return 'process_territory_region';
    case 'forced_elimination':
      return 'forced_elimination';
    default:
      return 'place_ring';
  }
}

// Helper to create mock move
// Note: The Move type in the component expects a 'type' property for getMoveTypeLabel
function createMockMove(phase: GamePhase, playerNumber: number): Move {
  return {
    id: 'move-1',
    type: getMoveTypeForPhase(phase),
    player: playerNumber,
    playerNumber,
    to: { x: 3, y: 3 },
    timestamp: new Date(),
    moveNumber: 1,
    thinkTime: 1000,
    phase,
  } as Move;
}

// Helper to create evaluation data
function createEvaluationData(
  moveNumber: number,
  totalEval: number
): PositionEvaluationPayload['data'] {
  return {
    gameId: 'test-game',
    moveNumber,
    boardType: 'square8',
    engineProfile: 'test',
    evaluationScale: 'zero_sum_margin',
    perPlayer: {
      1: {
        totalEval,
        territoryEval: totalEval * 0.5,
        ringEval: totalEval * 0.5,
      },
      2: {
        totalEval: -totalEval,
        territoryEval: -totalEval * 0.5,
        ringEval: -totalEval * 0.5,
      },
    },
  };
}

// Helper to create MoveAnalysis
function createMoveAnalysis(overrides: Partial<MoveAnalysis> = {}): MoveAnalysis {
  return {
    move: createMockMove('ring_placement', 1),
    moveNumber: 1,
    playerNumber: 1,
    evaluation: createEvaluationData(1, 5),
    prevEvaluation: createEvaluationData(0, 0),
    thinkTimeMs: 1500,
    engineDepth: 4,
    ...overrides,
  };
}

describe('MoveAnalysisPanel', () => {
  const players = createTestPlayers();

  describe('Empty State', () => {
    it('renders without crashing when analysis is null', () => {
      render(<MoveAnalysisPanel analysis={null} players={players} />);

      expect(screen.getByTestId('move-analysis-panel')).toBeInTheDocument();
    });

    it('displays instruction to select a move when analysis is null', () => {
      render(<MoveAnalysisPanel analysis={null} players={players} />);

      expect(screen.getByText('Select a move to see detailed analysis.')).toBeInTheDocument();
    });

    it('shows title in empty state', () => {
      render(<MoveAnalysisPanel analysis={null} players={players} />);

      expect(screen.getByText('Move Analysis')).toBeInTheDocument();
    });
  });

  describe('With Analysis Data', () => {
    it('renders player name and move number', () => {
      const analysis = createMoveAnalysis();

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Move #1')).toBeInTheDocument();
    });

    it('displays move type label', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('ring_placement', 1),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Type:')).toBeInTheDocument();
      expect(screen.getByText('Placement')).toBeInTheDocument();
    });

    it('displays position evaluation', () => {
      const analysis = createMoveAnalysis();

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Position Eval:')).toBeInTheDocument();
    });

    it('displays eval change with positive delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, 5),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Eval Change:')).toBeInTheDocument();
      // Both Position Eval and Eval Change show +5.0, so check they both exist
      const evalValues = screen.getAllByText('+5.0');
      expect(evalValues.length).toBeGreaterThanOrEqual(1);
    });

    it('displays eval change with negative delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, -3),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Eval Change:')).toBeInTheDocument();
      // Both Position Eval and Eval Change show -3.0, so check they both exist
      const evalValues = screen.getAllByText('-3.0');
      expect(evalValues.length).toBeGreaterThanOrEqual(1);
    });

    it('displays think time when provided', () => {
      const analysis = createMoveAnalysis({
        thinkTimeMs: 2500,
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Think Time:')).toBeInTheDocument();
      expect(screen.getByText('2.5s')).toBeInTheDocument();
    });

    it('displays engine depth when provided', () => {
      const analysis = createMoveAnalysis({
        engineDepth: 6,
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Search Depth:')).toBeInTheDocument();
      expect(screen.getByText('6')).toBeInTheDocument();
    });

    it('shows "no AI evaluation" message when evaluation is undefined', () => {
      const analysis = createMoveAnalysis({
        evaluation: undefined,
        prevEvaluation: undefined,
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('No AI evaluation available for this move.')).toBeInTheDocument();
    });
  });

  describe('Move Quality Classification', () => {
    it('shows "Excellent" badge for large positive delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, 10),
        prevEvaluation: createEvaluationData(1, 5),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Excellent')).toBeInTheDocument();
    });

    it('shows "Good" badge for moderate positive delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, 2),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Good')).toBeInTheDocument();
    });

    it('shows "Book" badge for neutral delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, 0.5),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Book')).toBeInTheDocument();
    });

    it('shows "Inaccuracy" badge for small negative delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, -2),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Inaccuracy')).toBeInTheDocument();
    });

    it('shows "Mistake" badge for moderate negative delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, -5),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Mistake')).toBeInTheDocument();
    });

    it('shows "Blunder" badge for large negative delta', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(2, -10),
        prevEvaluation: createEvaluationData(1, 0),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Blunder')).toBeInTheDocument();
    });
  });

  describe('Move Types', () => {
    it('displays "Movement" for movement phase', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('movement', 1),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Movement')).toBeInTheDocument();
    });

    it('displays "Capture" for capture phase', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('capture', 1),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      // overtaking_capture maps to 'Capture'
      expect(screen.getByText('Capture')).toBeInTheDocument();
    });

    it('displays "Capture" for chain capture phase', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('chain_capture', 1),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      // continue_capture_segment maps to 'Capture'
      expect(screen.getByText('Capture')).toBeInTheDocument();
    });

    it('displays "Line" for line processing phase', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('line_processing', 1),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      // process_line maps to 'Line'
      expect(screen.getByText('Line')).toBeInTheDocument();
    });

    it('displays "Territory" for territory processing phase', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('territory_processing', 1),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Territory')).toBeInTheDocument();
    });
  });

  describe('Think Time Formatting', () => {
    it('formats milliseconds correctly', () => {
      const analysis = createMoveAnalysis({
        thinkTimeMs: 500,
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('500ms')).toBeInTheDocument();
    });

    it('formats seconds correctly', () => {
      const analysis = createMoveAnalysis({
        thinkTimeMs: 5000,
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('5.0s')).toBeInTheDocument();
    });

    it('formats minutes correctly', () => {
      const analysis = createMoveAnalysis({
        thinkTimeMs: 90000,
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('1m 30s')).toBeInTheDocument();
    });
  });

  describe('Player Colors', () => {
    it('displays player indicator with color', () => {
      const analysis = createMoveAnalysis();

      const { container } = render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      // Should have a colored indicator element
      const colorIndicators = container.querySelectorAll('[aria-hidden="true"]');
      expect(colorIndicators.length).toBeGreaterThan(0);
    });

    it('handles player 2 correctly', () => {
      const analysis = createMoveAnalysis({
        move: createMockMove('movement', 2),
        playerNumber: 2,
        evaluation: createEvaluationData(1, -3),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Bob')).toBeInTheDocument();
    });
  });

  describe('Custom className', () => {
    it('applies custom className', () => {
      const analysis = createMoveAnalysis();

      render(<MoveAnalysisPanel analysis={analysis} players={players} className="custom-class" />);

      const panel = screen.getByTestId('move-analysis-panel');
      expect(panel).toHaveClass('custom-class');
    });
  });

  describe('Evaluation Breakdown', () => {
    it('shows territory and ring eval breakdown when available', () => {
      const analysis = createMoveAnalysis({
        evaluation: createEvaluationData(1, 6),
      });

      render(<MoveAnalysisPanel analysis={analysis} players={players} />);

      expect(screen.getByText('Breakdown:')).toBeInTheDocument();
    });
  });

  describe('Fallback Player Names', () => {
    it('shows fallback name when player username is empty', () => {
      const playersWithoutUsername: Player[] = [
        {
          id: 'p1',
          username: '',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const analysis = createMoveAnalysis();

      render(<MoveAnalysisPanel analysis={analysis} players={playersWithoutUsername} />);

      expect(screen.getByText('Player 1')).toBeInTheDocument();
    });
  });
});
