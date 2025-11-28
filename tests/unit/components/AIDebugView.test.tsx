import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AIDebugView } from '../../../src/client/components/AIDebugView';
import { GameState, BoardType } from '../../../src/shared/types/game';

// Helper to create a minimal GameState for testing
function createMinimalGameState(overrides: Partial<GameState> = {}): GameState {
  return {
    id: 'test-game-1',
    boardType: 'square8' as BoardType,
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    },
    players: [
      {
        id: 'player-1',
        username: 'Player 1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 300,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'player-2',
        username: 'AI Bot',
        type: 'ai',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 300,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
        aiProfile: {
          difficulty: 5,
          aiType: 'heuristic',
        },
      },
    ],
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 300, increment: 5, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 18,
    territoryVictoryThreshold: 32,
    ...overrides,
  };
}

describe('AIDebugView', () => {
  describe('conditional rendering', () => {
    it('returns null when no aiEvaluation and no aiThinking', () => {
      const gameState = createMinimalGameState();
      const { container } = render(<AIDebugView gameState={gameState} />);
      expect(container.firstChild).toBeNull();
    });

    it('returns null when aiEvaluation is undefined and aiThinking is false', () => {
      const gameState = createMinimalGameState();
      const { container } = render(
        <AIDebugView gameState={gameState} aiEvaluation={undefined} aiThinking={false} />
      );
      expect(container.firstChild).toBeNull();
    });

    it('renders when aiThinking is true even without aiEvaluation', () => {
      const gameState = createMinimalGameState();
      const { container } = render(<AIDebugView gameState={gameState} aiThinking={true} />);
      expect(container.firstChild).not.toBeNull();
    });

    it('renders when aiEvaluation is provided', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1.5, breakdown: { material: 1.0, position: 0.5 } };
      const { container } = render(
        <AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />
      );
      expect(container.firstChild).not.toBeNull();
    });
  });

  describe('header and title', () => {
    it('displays "AI Analysis" header', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('AI Analysis')).toBeInTheDocument();
    });

    it('shows "Thinking..." indicator when aiThinking is true', () => {
      const gameState = createMinimalGameState();
      render(<AIDebugView gameState={gameState} aiThinking={true} />);
      expect(screen.getByText('Thinking...')).toBeInTheDocument();
    });

    it('does not show "Thinking..." indicator when aiThinking is false', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} aiThinking={false} />);
      expect(screen.queryByText('Thinking...')).not.toBeInTheDocument();
    });

    it('applies animate-pulse to "Thinking..." indicator', () => {
      const gameState = createMinimalGameState();
      render(<AIDebugView gameState={gameState} aiThinking={true} />);
      const thinkingElement = screen.getByText('Thinking...');
      expect(thinkingElement).toHaveClass('animate-pulse');
    });
  });

  describe('score display', () => {
    it('displays total score value', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 2.5, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('Total Score')).toBeInTheDocument();
      expect(screen.getByText('+2.50')).toBeInTheDocument();
    });

    it('displays positive score with + prefix and green color', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 3.14, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      const scoreElement = screen.getByText('+3.14');
      expect(scoreElement).toHaveClass('text-emerald-400');
    });

    it('displays negative score without + prefix and red color', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: -1.5, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      const scoreElement = screen.getByText('-1.50');
      expect(scoreElement).toHaveClass('text-red-400');
    });

    it('displays zero score with neutral color', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      const scoreElement = screen.getByText('0.00');
      expect(scoreElement).toHaveClass('text-slate-200');
    });

    it('formats score to 2 decimal places', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1.333333, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('+1.33')).toBeInTheDocument();
    });
  });

  describe('breakdown display', () => {
    it('displays "Breakdown" section label', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1, breakdown: { material: 0.5 } };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('Breakdown')).toBeInTheDocument();
    });

    it('displays breakdown entries with formatted names', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1.5, breakdown: { ring_control: 1.0, territory_bonus: 0.5 } };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('ring control')).toBeInTheDocument();
      expect(screen.getByText('territory bonus')).toBeInTheDocument();
    });

    it('displays breakdown values with correct formatting', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1.5, breakdown: { material: 1.25, position: -0.75 } };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('+1.25')).toBeInTheDocument();
      expect(screen.getByText('-0.75')).toBeInTheDocument();
    });

    it('excludes "total" key from breakdown display', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1.5, breakdown: { total: 1.5, material: 1.0, position: 0.5 } };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('material')).toBeInTheDocument();
      expect(screen.getByText('position')).toBeInTheDocument();
      // The "total" should only appear once (in the Total Score label), not in breakdown
      const totalElements = screen.getAllByText(/total/i);
      expect(totalElements).toHaveLength(1);
    });

    it('handles empty breakdown object', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText('Breakdown')).toBeInTheDocument();
    });
  });

  describe('AI type and difficulty display', () => {
    it('displays AI type for current player', () => {
      const gameState = createMinimalGameState({ currentPlayer: 2 });
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/AI Type:/)).toBeInTheDocument();
      expect(screen.getByText(/heuristic/)).toBeInTheDocument();
    });

    it('displays difficulty for current player', () => {
      const gameState = createMinimalGameState({ currentPlayer: 2 });
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/Difficulty:/)).toBeInTheDocument();
      expect(screen.getByText(/5/)).toBeInTheDocument();
    });

    it('displays "Unknown" for AI type when player has no aiProfile', () => {
      const gameState = createMinimalGameState({ currentPlayer: 1 });
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/Unknown/)).toBeInTheDocument();
    });

    it('displays "?" for difficulty when player has no aiProfile', () => {
      const gameState = createMinimalGameState({ currentPlayer: 1 });
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/\?/)).toBeInTheDocument();
    });

    it('handles different AI types (minimax)', () => {
      const gameState = createMinimalGameState();
      gameState.players[1].aiProfile!.aiType = 'minimax';
      gameState.currentPlayer = 2;
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/minimax/)).toBeInTheDocument();
    });

    it('handles different AI types (mcts)', () => {
      const gameState = createMinimalGameState();
      gameState.players[1].aiProfile!.aiType = 'mcts';
      gameState.currentPlayer = 2;
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/mcts/)).toBeInTheDocument();
    });

    it('handles different difficulty levels', () => {
      const gameState = createMinimalGameState();
      gameState.players[1].aiProfile!.difficulty = 10;
      gameState.currentPlayer = 2;
      const aiEvaluation = { score: 0, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      expect(screen.getByText(/10/)).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies correct container styling', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 0, breakdown: {} };
      const { container } = render(
        <AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />
      );
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('p-4', 'border', 'border-slate-700', 'rounded-2xl');
    });

    it('renders score with font-mono styling', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 1.5, breakdown: {} };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} />);
      const scoreElement = screen.getByText('+1.50');
      expect(scoreElement).toHaveClass('font-mono');
    });
  });

  describe('combined states', () => {
    it('shows both thinking indicator and evaluation when both provided', () => {
      const gameState = createMinimalGameState();
      const aiEvaluation = { score: 2.5, breakdown: { material: 1.5, position: 1.0 } };
      render(<AIDebugView gameState={gameState} aiEvaluation={aiEvaluation} aiThinking={true} />);
      expect(screen.getByText('Thinking...')).toBeInTheDocument();
      expect(screen.getByText('+2.50')).toBeInTheDocument();
      expect(screen.getByText('material')).toBeInTheDocument();
      expect(screen.getByText('position')).toBeInTheDocument();
    });
  });
});
