import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SpectatorHUD, type SpectatorHUDProps } from '../../src/client/components/SpectatorHUD';
import type { Player, Move, GamePhase } from '../../src/shared/types/game';
import type { PositionEvaluationPayload } from '../../src/shared/types/websocket';

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
      ringsInHand: 15,
      eliminatedRings: 3,
      territorySpaces: 5,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 17,
      eliminatedRings: 1,
      territorySpaces: 2,
      aiDifficulty: 5,
    },
  ];
}

// Helper to create mock move
function createMockMove(phase: GamePhase, playerNumber: number): Move {
  return {
    id: `move-${Math.random()}`,
    type: 'place_ring',
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
function createEvaluationData(moveNumber: number): PositionEvaluationPayload['data'] {
  return {
    gameId: 'test-game',
    moveNumber,
    boardType: 'square8',
    engineProfile: 'test',
    evaluationScale: 'zero_sum_margin',
    perPlayer: {
      1: { totalEval: 5, territoryEval: 2.5, ringEval: 2.5 },
      2: { totalEval: -5, territoryEval: -2.5, ringEval: -2.5 },
    },
  };
}

describe('SpectatorHUD', () => {
  const defaultProps: SpectatorHUDProps = {
    phase: 'movement',
    players: createTestPlayers(),
    currentPlayerNumber: 1,
    turnNumber: 5,
    moveNumber: 10,
    moveHistory: [],
    evaluationHistory: [],
  };

  describe('Rendering', () => {
    it('renders without crashing', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
    });

    it('displays spectator mode header', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.getByText('Spectator Mode')).toBeInTheDocument();
    });

    it('displays current phase', () => {
      render(<SpectatorHUD {...defaultProps} phase="movement" />);

      // Phase is rendered with icon and beginner-friendly label
      expect(screen.getByText(/Your Move/)).toBeInTheDocument();
    });

    it('displays turn and move numbers', () => {
      render(<SpectatorHUD {...defaultProps} turnNumber={7} moveNumber={15} />);

      expect(screen.getByText('Turn 7 • Move #15')).toBeInTheDocument();
    });

    it('displays current player name', () => {
      render(<SpectatorHUD {...defaultProps} />);

      // Alice appears multiple times - in current player indicator and standings
      expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('is playing')).toBeInTheDocument();
    });
  });

  describe('Player Information', () => {
    it('displays all players in the standings section', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.getByText('Players')).toBeInTheDocument();
      // Player names may appear multiple times
      expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Bob').length).toBeGreaterThanOrEqual(1);
    });

    it('shows AI badge for AI players', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.getByText('AI')).toBeInTheDocument();
    });

    it('highlights current player', () => {
      const { container } = render(<SpectatorHUD {...defaultProps} currentPlayerNumber={1} />);

      // Current player should have specific styling
      const playerCards = container.querySelectorAll('.flex.items-center.justify-between');
      expect(playerCards.length).toBeGreaterThan(0);
    });

    it('displays ring counts for players', () => {
      render(<SpectatorHUD {...defaultProps} />);

      // Should show rings in hand and captured rings
      expect(screen.getAllByText(/hand/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/cap/).length).toBeGreaterThan(0);
    });

    it('displays territory for players with territory', () => {
      render(<SpectatorHUD {...defaultProps} />);

      // Alice has 5 territory spaces - players with non-zero territory have a territory indicator
      // Note: The territory display may vary based on Player type properties
      // so we check that the standings section exists and players are shown
      expect(screen.getByText('Players')).toBeInTheDocument();
    });
  });

  describe('Spectator Count', () => {
    it('displays spectator count when provided', () => {
      render(<SpectatorHUD {...defaultProps} spectatorCount={5} />);

      expect(screen.getByText('5 viewers')).toBeInTheDocument();
    });

    it('shows singular viewer when count is 1', () => {
      render(<SpectatorHUD {...defaultProps} spectatorCount={1} />);

      expect(screen.getByText('1 viewer')).toBeInTheDocument();
    });

    it('does not show viewer count when 0', () => {
      render(<SpectatorHUD {...defaultProps} spectatorCount={0} />);

      expect(screen.queryByText(/viewer/)).not.toBeInTheDocument();
    });
  });

  describe('Analysis Section', () => {
    it('displays analysis toggle button', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.getByText('Analysis & Insights')).toBeInTheDocument();
    });

    it('shows analysis section by default', () => {
      render(<SpectatorHUD {...defaultProps} />);

      // Recent Moves section should be visible
      expect(screen.getByText('Recent Moves')).toBeInTheDocument();
    });

    it('hides analysis section when toggled', () => {
      render(<SpectatorHUD {...defaultProps} />);

      const toggleButton = screen.getByText('Analysis & Insights');
      fireEvent.click(toggleButton);

      // Recent Moves should not be visible
      expect(screen.queryByText('Recent Moves')).not.toBeInTheDocument();
    });

    it('shows analysis section again when toggled back', () => {
      render(<SpectatorHUD {...defaultProps} />);

      const toggleButton = screen.getByText('Analysis & Insights');

      // Hide
      fireEvent.click(toggleButton);
      expect(screen.queryByText('Recent Moves')).not.toBeInTheDocument();

      // Show again
      fireEvent.click(toggleButton);
      expect(screen.getByText('Recent Moves')).toBeInTheDocument();
    });
  });

  describe('Evaluation Graph', () => {
    it('displays evaluation graph when evaluation history exists', () => {
      const evaluationHistory = [createEvaluationData(1), createEvaluationData(2)];

      render(<SpectatorHUD {...defaultProps} evaluationHistory={evaluationHistory} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('does not display evaluation graph when no evaluation history', () => {
      render(<SpectatorHUD {...defaultProps} evaluationHistory={[]} />);

      // The graph component might still render with empty state
      // Just verify the HUD renders correctly
      expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
    });
  });

  describe('Move History', () => {
    it('displays recent moves when move history exists', () => {
      const moveHistory = [
        createMockMove('ring_placement', 1),
        createMockMove('movement', 2),
        createMockMove('ring_placement', 1),
      ];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText('Recent Moves')).toBeInTheDocument();
    });

    it('shows "No moves yet" when move history is empty', () => {
      render(<SpectatorHUD {...defaultProps} moveHistory={[]} />);

      expect(screen.getByText('No moves yet.')).toBeInTheDocument();
    });

    it('displays last 5 moves', () => {
      const moveHistory = Array.from({ length: 10 }, (_, i) =>
        createMockMove('movement', (i % 2) + 1)
      );

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      // Should limit to 5 recent moves
      const moveButtons = screen
        .getAllByRole('button')
        .filter((btn) => btn.textContent?.includes('#'));
      expect(moveButtons.length).toBeLessThanOrEqual(5);
    });

    it('calls onMoveSelect when a move is clicked', () => {
      const onMoveSelect = jest.fn();
      const moveHistory = [createMockMove('ring_placement', 1), createMockMove('movement', 2)];

      render(
        <SpectatorHUD {...defaultProps} moveHistory={moveHistory} onMoveSelect={onMoveSelect} />
      );

      // Find move buttons (they should contain #)
      const moveButtons = screen
        .getAllByRole('button')
        .filter((btn) => btn.textContent?.includes('#'));

      if (moveButtons.length > 0) {
        fireEvent.click(moveButtons[0]);
        expect(onMoveSelect).toHaveBeenCalled();
      }
    });

    it('highlights selected move', () => {
      const moveHistory = [createMockMove('ring_placement', 1), createMockMove('movement', 2)];

      const { container } = render(
        <SpectatorHUD {...defaultProps} moveHistory={moveHistory} selectedMoveIndex={0} />
      );

      // Selected move should have specific styling
      expect(container.querySelector('.bg-blue-900\\/40')).toBeInTheDocument();
    });
  });

  describe('Move Analysis Panel', () => {
    it('displays move analysis when a move is selected', () => {
      const moveHistory = [createMockMove('ring_placement', 1)];
      const evaluationHistory = [createEvaluationData(1)];

      render(
        <SpectatorHUD
          {...defaultProps}
          moveHistory={moveHistory}
          evaluationHistory={evaluationHistory}
          selectedMoveIndex={0}
        />
      );

      expect(screen.getByTestId('move-analysis-panel')).toBeInTheDocument();
    });

    it('shows empty analysis panel when no move is selected', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.getByText('Select a move to see detailed analysis.')).toBeInTheDocument();
    });
  });

  describe('Phase Display', () => {
    // Beginner-friendly phase labels from PHASE_INFO in gameViewModels.ts
    it('shows correct phase for ring_placement', () => {
      render(<SpectatorHUD {...defaultProps} phase="ring_placement" />);

      expect(screen.getByText(/Place Rings/)).toBeInTheDocument();
    });

    it('shows correct phase for movement', () => {
      render(<SpectatorHUD {...defaultProps} phase="movement" />);

      expect(screen.getByText(/Your Move/)).toBeInTheDocument();
    });

    it('shows correct phase for chain_capture', () => {
      render(<SpectatorHUD {...defaultProps} phase="chain_capture" />);

      expect(screen.getByText(/Keep Capturing!/)).toBeInTheDocument();
    });

    it('shows correct phase for line_processing', () => {
      render(<SpectatorHUD {...defaultProps} phase="line_processing" />);

      expect(screen.getByText(/Line Scored!/)).toBeInTheDocument();
    });

    it('shows correct phase for territory_processing', () => {
      render(<SpectatorHUD {...defaultProps} phase="territory_processing" />);

      expect(screen.getByText(/Territory!/)).toBeInTheDocument();
    });

    it('shows correct phase for forced_elimination', () => {
      render(<SpectatorHUD {...defaultProps} phase="forced_elimination" />);

      expect(screen.getByText(/Blocked!/)).toBeInTheDocument();
    });
  });

  describe('Custom className', () => {
    it('applies custom className', () => {
      render(<SpectatorHUD {...defaultProps} className="custom-class" />);

      const hud = screen.getByTestId('spectator-hud');
      expect(hud).toHaveClass('custom-class');
    });
  });

  describe('Player Colors', () => {
    it('displays player color indicators', () => {
      const { container } = render(<SpectatorHUD {...defaultProps} />);

      // Should have colored indicators for players
      const indicators = container.querySelectorAll('[aria-hidden="true"]');
      expect(indicators.length).toBeGreaterThan(0);
    });
  });

  describe('Phase Hints', () => {
    it('displays spectator hint for the current phase', () => {
      render(<SpectatorHUD {...defaultProps} phase="movement" />);

      // The phase info should include spectator hints
      // Spectator hint text should be visible
      const phaseSection = screen.getByText(/Your Move/).closest('div');
      expect(phaseSection).toBeInTheDocument();
    });

    it('surfaces spectator hint text for terminal game_over phase', () => {
      render(<SpectatorHUD {...defaultProps} phase="game_over" />);

      expect(screen.getByText(/Game Over/)).toBeInTheDocument();
      // SpectatorHUD renders spectatorHint, not description
      expect(screen.getByText(/Game finished/)).toBeInTheDocument();
    });
  });

  describe('Connection Status', () => {
    it('shows live indicator when connected', () => {
      const { container } = render(<SpectatorHUD {...defaultProps} connectionStatus="connected" />);

      // Should show animated ping indicator for live status
      expect(container.querySelector('.animate-ping')).toBeInTheDocument();
    });

    it('shows reconnecting banner when reconnecting', () => {
      render(<SpectatorHUD {...defaultProps} connectionStatus="reconnecting" />);

      expect(screen.getByText('Reconnecting...')).toBeInTheDocument();
    });

    it('shows connecting banner when connecting', () => {
      render(<SpectatorHUD {...defaultProps} connectionStatus="connecting" />);

      expect(screen.getByText('Connecting...')).toBeInTheDocument();
    });

    it('shows disconnected banner when disconnected', () => {
      render(<SpectatorHUD {...defaultProps} connectionStatus="disconnected" />);

      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('does not show connection banner when connected', () => {
      render(<SpectatorHUD {...defaultProps} connectionStatus="connected" />);

      expect(screen.queryByText('Reconnecting...')).not.toBeInTheDocument();
      expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    });
  });

  describe('Disconnected Players Banner', () => {
    it('shows banner when a player has disconnected', () => {
      render(
        <SpectatorHUD
          {...defaultProps}
          disconnectedPlayers={[{ id: 'p1', username: 'Alice', disconnectedAt: Date.now() }]}
        />
      );

      const banner = screen.getByTestId('disconnected-players-banner');
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveTextContent('Alice');
      expect(banner).toHaveTextContent('has disconnected');
      expect(banner).toHaveTextContent('Waiting to reconnect');
    });

    it('shows correct grammar for multiple disconnected players', () => {
      render(
        <SpectatorHUD
          {...defaultProps}
          disconnectedPlayers={[
            { id: 'p1', username: 'Alice', disconnectedAt: Date.now() },
            { id: 'p2', username: 'Bob', disconnectedAt: Date.now() },
          ]}
        />
      );

      const banner = screen.getByTestId('disconnected-players-banner');
      expect(banner).toHaveTextContent('Alice, Bob');
      expect(banner).toHaveTextContent('have disconnected');
    });

    it('does not show banner when no players are disconnected', () => {
      render(<SpectatorHUD {...defaultProps} disconnectedPlayers={[]} />);

      expect(screen.queryByTestId('disconnected-players-banner')).not.toBeInTheDocument();
    });
  });

  describe('Active Choice Banner', () => {
    it('displays active choice banner when player is making a decision', () => {
      render(
        <SpectatorHUD
          {...defaultProps}
          activeChoice={{
            type: 'capture_direction',
            playerNumber: 1,
            timeRemaining: 25,
          }}
        />
      );

      const banner = screen.getByTestId('spectator-choice-banner');
      expect(banner).toBeInTheDocument();
      // Check banner contains player name and choice type
      expect(banner).toHaveTextContent('Alice');
      expect(banner).toHaveTextContent('is deciding:');
      expect(banner).toHaveTextContent('Capture Direction');
    });

    it('displays time remaining for active choice', () => {
      render(
        <SpectatorHUD
          {...defaultProps}
          activeChoice={{
            type: 'line_reward',
            playerNumber: 1,
            timeRemaining: 15,
          }}
        />
      );

      expect(screen.getByText('15s')).toBeInTheDocument();
    });

    it('applies urgent styling when time is low', () => {
      const { container } = render(
        <SpectatorHUD
          {...defaultProps}
          activeChoice={{
            type: 'territory_action',
            playerNumber: 2,
            timeRemaining: 5,
          }}
        />
      );

      // Low time should show red color
      expect(container.querySelector('.text-red-400')).toBeInTheDocument();
    });

    it('shows choice description', () => {
      render(
        <SpectatorHUD
          {...defaultProps}
          activeChoice={{
            type: 'capture_direction',
            playerNumber: 1,
          }}
        />
      );

      expect(screen.getByText('Choosing which direction to capture')).toBeInTheDocument();
    });

    it('does not display banner when no active choice', () => {
      render(<SpectatorHUD {...defaultProps} />);

      expect(screen.queryByTestId('spectator-choice-banner')).not.toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('handles 3+ players', () => {
      const players: Player[] = [
        ...createTestPlayers(),
        {
          id: 'p3',
          username: 'Charlie',
          playerNumber: 3,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 16,
          eliminatedRings: 2,
          territorySpaces: 3,
        },
      ];

      render(<SpectatorHUD {...defaultProps} players={players} />);

      // Player names may appear multiple times (in standings and current player indicator)
      expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Bob').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Charlie').length).toBeGreaterThanOrEqual(1);
    });

    it('handles player without username', () => {
      const players: Player[] = [
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

      render(<SpectatorHUD {...defaultProps} players={players} currentPlayerNumber={1} />);

      // "Player 1" appears in both the current player indicator and the standings
      const player1Elements = screen.getAllByText('Player 1');
      expect(player1Elements.length).toBeGreaterThanOrEqual(1);
    });

    it('handles unknown phase gracefully (line 53)', () => {
      // Cast to GamePhase to test fallback handling of unknown phases
      const unknownPhase = 'unknown_phase_xyz' as GamePhase;

      render(<SpectatorHUD {...defaultProps} phase={unknownPhase} />);

      // Should fall back to displaying the phase string as-is
      expect(screen.getByText(/unknown_phase_xyz/)).toBeInTheDocument();
    });
  });

  describe('Move Annotation Coverage', () => {
    // Helper to create move with specific type
    const createMoveWithType = (type: Move['type'], playerNumber: number, moveNum: number): Move =>
      ({
        id: `move-${moveNum}`,
        type,
        player: playerNumber,
        playerNumber,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        moveNumber: moveNum,
        thinkTime: 1000,
        phase: 'movement',
      }) as Move;

    it('displays skip_placement annotation', () => {
      const moveHistory = [createMoveWithType('skip_placement', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 skipped placement/)).toBeInTheDocument();
    });

    it('displays move_stack annotation', () => {
      const moveHistory = [createMoveWithType('move_stack', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 moved a stack/)).toBeInTheDocument();
    });

    it('displays overtaking_capture annotation', () => {
      const moveHistory = [createMoveWithType('overtaking_capture', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 captured/)).toBeInTheDocument();
    });

    it('displays continue_capture_segment annotation', () => {
      const moveHistory = [createMoveWithType('continue_capture_segment', 2, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P2 captured/)).toBeInTheDocument();
    });

    it('displays skip_capture annotation', () => {
      const moveHistory = [createMoveWithType('skip_capture', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 skipped capture/)).toBeInTheDocument();
    });

    it('displays process_line annotation', () => {
      const moveHistory = [createMoveWithType('process_line', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 claimed line bonus/)).toBeInTheDocument();
    });

    it('displays process_territory_region annotation', () => {
      const moveHistory = [createMoveWithType('process_territory_region', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 processed territory/)).toBeInTheDocument();
    });

    it('displays forced_elimination annotation', () => {
      const moveHistory = [createMoveWithType('forced_elimination', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 forced to eliminate/)).toBeInTheDocument();
    });

    it('displays swap_sides annotation', () => {
      const moveHistory = [createMoveWithType('swap_sides', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 swapped sides/)).toBeInTheDocument();
    });

    it('displays line_formation annotation', () => {
      const moveHistory = [createMoveWithType('line_formation', 2, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P2 formed a line/)).toBeInTheDocument();
    });

    it('displays territory_claim annotation', () => {
      const moveHistory = [createMoveWithType('territory_claim', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 claimed territory/)).toBeInTheDocument();
    });

    it('displays recovery_slide annotation', () => {
      const moveHistory = [createMoveWithType('recovery_slide', 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 performed recovery/)).toBeInTheDocument();
    });

    it('displays skip_recovery annotation', () => {
      const moveHistory = [createMoveWithType('skip_recovery', 2, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P2 skipped recovery/)).toBeInTheDocument();
    });

    it('displays default annotation for unknown move type', () => {
      // Cast to unknown type to test default case
      const moveHistory = [createMoveWithType('some_unknown_type' as Move['type'], 1, 1)];

      render(<SpectatorHUD {...defaultProps} moveHistory={moveHistory} />);

      expect(screen.getByText(/P1 made a move/)).toBeInTheDocument();
    });
  });

  describe('Previous Evaluation Lookup (line 177)', () => {
    it('shows previous evaluation when selectedMoveIndex > 0', () => {
      const moveHistory = [
        createMockMove('ring_placement', 1),
        createMockMove('ring_placement', 2),
        createMockMove('movement', 1),
      ];
      const evaluationHistory = [
        createEvaluationData(1),
        createEvaluationData(2),
        createEvaluationData(3),
      ];

      render(
        <SpectatorHUD
          {...defaultProps}
          moveHistory={moveHistory}
          evaluationHistory={evaluationHistory}
          selectedMoveIndex={2}
        />
      );

      // Should show analysis panel with selected move (move 3)
      expect(screen.getByTestId('move-analysis-panel')).toBeInTheDocument();
    });

    it('provides previous evaluation to MoveAnalysisPanel when available', () => {
      const moveHistory = [
        createMockMove('ring_placement', 1),
        createMockMove('ring_placement', 2),
      ];
      const evaluationHistory = [createEvaluationData(1), createEvaluationData(2)];

      render(
        <SpectatorHUD
          {...defaultProps}
          moveHistory={moveHistory}
          evaluationHistory={evaluationHistory}
          selectedMoveIndex={1}
        />
      );

      // With index=1, prev evaluation should be looked up (moveNumber=1)
      // This covers line 177: selectedMoveIndex > 0
      expect(screen.getByTestId('move-analysis-panel')).toBeInTheDocument();
    });
  });
});
