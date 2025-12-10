import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SpectatorHUD } from '../../../src/client/components/SpectatorHUD';
import type { Player, Move, GamePhase } from '../../../src/shared/types/game';
import type { PositionEvaluationPayload } from '../../../src/shared/types/websocket';

// Mock child components that have complex dependencies
jest.mock('../../../src/client/components/EvaluationGraph', () => {
  const React = require('react');
  return {
    EvaluationGraph: function MockEvaluationGraph({
      evaluationHistory,
    }: {
      evaluationHistory: unknown[];
    }) {
      return React.createElement(
        'div',
        { 'data-testid': 'evaluation-graph' },
        `EvaluationGraph (${evaluationHistory.length} evals)`
      );
    },
  };
});

jest.mock('../../../src/client/components/MoveAnalysisPanel', () => {
  const React = require('react');
  return {
    MoveAnalysisPanel: function MockMoveAnalysisPanel({ analysis }: { analysis: unknown }) {
      return React.createElement(
        'div',
        { 'data-testid': 'move-analysis-panel' },
        `MoveAnalysisPanel ${analysis ? '(with analysis)' : '(empty)'}`
      );
    },
  };
});

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 120_000,
      ringsInHand: 5,
      eliminatedRings: 1,
      territorySpaces: 2,
      totalRings: 24,
      ringsEliminated: 1,
      territory: 2,
    },
    {
      id: 'p2',
      username: 'BotPlayer',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 90_000,
      ringsInHand: 4,
      eliminatedRings: 2,
      territorySpaces: 0,
      totalRings: 24,
      ringsEliminated: 2,
      territory: 0,
    },
  ];
}

function createMoveHistory(): Move[] {
  return [
    {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      playerNumber: 1,
      phase: 'ring_placement',
      from: null,
      to: { x: 0, y: 0, z: null },
      timestamp: new Date(),
    },
    {
      id: 'm2',
      type: 'place_ring',
      player: 2,
      playerNumber: 2,
      phase: 'ring_placement',
      from: null,
      to: { x: 1, y: 1, z: null },
      timestamp: new Date(),
    },
    {
      id: 'm3',
      type: 'move_stack',
      player: 1,
      playerNumber: 1,
      phase: 'movement',
      from: { x: 0, y: 0, z: null },
      to: { x: 0, y: 1, z: null },
      timestamp: new Date(),
    },
  ] as Move[];
}

function createEvaluationHistory(): PositionEvaluationPayload['data'][] {
  return [
    {
      moveNumber: 1,
      playerScores: { 1: 0.5, 2: 0.5 },
      confidence: 0.8,
      features: {},
    },
    {
      moveNumber: 2,
      playerScores: { 1: 0.52, 2: 0.48 },
      confidence: 0.85,
      features: {},
    },
  ];
}

describe('SpectatorHUD', () => {
  it('renders spectator mode header', () => {
    render(
      <SpectatorHUD
        phase="ring_placement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    expect(screen.getByText('Spectator Mode')).toBeInTheDocument();
    expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
  });

  it('displays spectator count when provided', () => {
    render(
      <SpectatorHUD
        phase="ring_placement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        spectatorCount={5}
      />
    );

    expect(screen.getByText('5 viewers')).toBeInTheDocument();
  });

  it('uses singular "viewer" for spectatorCount of 1', () => {
    render(
      <SpectatorHUD
        phase="ring_placement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        spectatorCount={1}
      />
    );

    expect(screen.getByText('1 viewer')).toBeInTheDocument();
  });

  it('displays current phase and turn info', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={3}
        moveNumber={7}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    expect(screen.getByText(/Turn 3/)).toBeInTheDocument();
    expect(screen.getByText(/Move #7/)).toBeInTheDocument();
  });

  it('shows current player name and "is playing" indicator', () => {
    const players = createPlayers();
    render(
      <SpectatorHUD
        phase="movement"
        players={players}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // Alice appears in both current player indicator and players list
    expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('is playing')).toBeInTheDocument();
  });

  it('displays all players with their stats', () => {
    const players = createPlayers();
    render(
      <SpectatorHUD
        phase="movement"
        players={players}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // Alice appears in both current player indicator and players list
    expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('BotPlayer')).toBeInTheDocument();
    // AI badge for player 2
    expect(screen.getByText('AI')).toBeInTheDocument();
  });

  it('shows "No moves yet" when moveHistory is empty', () => {
    render(
      <SpectatorHUD
        phase="ring_placement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    expect(screen.getByText('No moves yet.')).toBeInTheDocument();
  });

  it('renders recent moves from moveHistory', () => {
    const moveHistory = createMoveHistory();
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={4}
        moveHistory={moveHistory}
        evaluationHistory={[]}
      />
    );

    // Recent moves section should show move annotations
    expect(screen.getByText(/P1 placed a ring/)).toBeInTheDocument();
    expect(screen.getByText(/P2 placed a ring/)).toBeInTheDocument();
    expect(screen.getByText(/P1 moved a stack/)).toBeInTheDocument();
  });

  it('renders EvaluationGraph when evaluationHistory is provided', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={3}
        moveHistory={createMoveHistory()}
        evaluationHistory={createEvaluationHistory()}
      />
    );

    expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    expect(screen.getByText(/EvaluationGraph \(2 evals\)/)).toBeInTheDocument();
  });

  it('renders MoveAnalysisPanel', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={3}
        moveHistory={createMoveHistory()}
        evaluationHistory={[]}
      />
    );

    expect(screen.getByTestId('move-analysis-panel')).toBeInTheDocument();
  });

  it('toggles analysis section when button is clicked', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={createMoveHistory()}
        evaluationHistory={createEvaluationHistory()}
      />
    );

    // Analysis should be visible by default
    expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();

    // Click the toggle button
    const toggleButton = screen.getByRole('button', { name: /Analysis & Insights/i });
    fireEvent.click(toggleButton);

    // Analysis should be hidden
    expect(screen.queryByTestId('evaluation-graph')).not.toBeInTheDocument();

    // Click again to show
    fireEvent.click(toggleButton);
    expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
  });

  it('calls onMoveSelect when a move is clicked in recent moves', () => {
    const onMoveSelect = jest.fn();
    const moveHistory = createMoveHistory();

    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={4}
        moveHistory={moveHistory}
        evaluationHistory={[]}
        onMoveSelect={onMoveSelect}
      />
    );

    // Click on the first move button (#1)
    const moveButton = screen.getByRole('button', { name: /#1.*P1 placed a ring/ });
    fireEvent.click(moveButton);

    expect(onMoveSelect).toHaveBeenCalledWith(0);
  });

  it('highlights selected move in recent moves list', () => {
    const moveHistory = createMoveHistory();

    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={4}
        moveHistory={moveHistory}
        evaluationHistory={[]}
        selectedMoveIndex={0}
      />
    );

    // The first move should have the selected styling (blue border)
    const moveButton = screen.getByRole('button', { name: /#1.*P1 placed a ring/ });
    expect(moveButton).toHaveClass('bg-blue-900/40');
  });

  it('applies custom className', () => {
    render(
      <SpectatorHUD
        phase="ring_placement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        className="custom-class"
      />
    );

    expect(screen.getByTestId('spectator-hud')).toHaveClass('custom-class');
  });

  it('shows territory stats when player has territory', () => {
    const players = createPlayers();
    // Player 1 has territory: 2
    render(
      <SpectatorHUD
        phase="movement"
        players={players}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // Territory stat should be visible
    expect(screen.getByText('2 terr')).toBeInTheDocument();
  });
});

describe('SpectatorHUD - Connection Status', () => {
  it('shows live indicator when connected', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        connectionStatus="connected"
      />
    );

    // Should have the animated live dot (emerald colored)
    const liveIndicator = screen.getByTestId('spectator-hud').querySelector('.animate-ping');
    expect(liveIndicator).toBeInTheDocument();
  });

  it('shows reconnecting banner when reconnecting', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        connectionStatus="reconnecting"
      />
    );

    expect(screen.getByText('Reconnecting...')).toBeInTheDocument();
  });

  it('shows disconnected banner when disconnected', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        connectionStatus="disconnected"
      />
    );

    expect(screen.getByText('Disconnected')).toBeInTheDocument();
  });

  it('shows connecting banner when connecting', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        connectionStatus="connecting"
      />
    );

    expect(screen.getByText('Connecting...')).toBeInTheDocument();
  });
});

describe('SpectatorHUD - Active Choice Banner', () => {
  it('shows choice banner when activeChoice is provided', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        activeChoice={{
          type: 'capture_direction',
          playerNumber: 1,
        }}
      />
    );

    expect(screen.getByTestId('spectator-choice-banner')).toBeInTheDocument();
    expect(screen.getByText('is deciding:')).toBeInTheDocument();
    expect(screen.getByText('Capture Direction')).toBeInTheDocument();
  });

  it('shows player name in choice banner', () => {
    const players = createPlayers();
    render(
      <SpectatorHUD
        phase="movement"
        players={players}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        activeChoice={{
          type: 'line_reward',
          playerNumber: 1,
        }}
      />
    );

    // Alice is player 1 - appears in both choice banner and player list
    const aliceElements = screen.getAllByText('Alice');
    expect(aliceElements.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Line Reward')).toBeInTheDocument();
  });

  it('shows countdown timer when timeRemaining is provided', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        activeChoice={{
          type: 'territory_action',
          playerNumber: 2,
          timeRemaining: 25,
        }}
      />
    );

    expect(screen.getByText('25s')).toBeInTheDocument();
  });

  it('shows urgency styling when time is low', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        activeChoice={{
          type: 'elimination_choice',
          playerNumber: 1,
          timeRemaining: 5,
        }}
      />
    );

    const timer = screen.getByText('5s');
    expect(timer).toHaveClass('text-red-400');
  });

  it('does not show choice banner when activeChoice is undefined', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    expect(screen.queryByTestId('spectator-choice-banner')).not.toBeInTheDocument();
  });
});
