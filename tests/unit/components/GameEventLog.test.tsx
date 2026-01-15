import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameEventLog, GameEventLogLegacyProps } from '../../../src/client/components/GameEventLog';
import {
  GameHistoryEntry,
  GameResult,
  Move,
  GamePhase,
  GameStatus,
  ProgressSnapshot,
} from '../../../src/shared/types/game';
import type {
  EventLogViewModel,
  EventLogItemViewModel,
} from '../../../src/client/adapters/gameViewModels';

// Helper to create a minimal Move
function createMove(overrides: Partial<Move> = {}): Move {
  return {
    id: 'move-1',
    type: 'place_ring',
    player: 1,
    to: { x: 3, y: 3 },
    timestamp: new Date(),
    thinkTime: 1000,
    moveNumber: 1,
    ...overrides,
  };
}

// Helper to create a GameHistoryEntry
function createHistoryEntry(overrides: Partial<GameHistoryEntry> = {}): GameHistoryEntry {
  return {
    moveNumber: 1,
    action: createMove(),
    actor: 1,
    phaseBefore: 'ring_placement' as GamePhase,
    phaseAfter: 'ring_placement' as GamePhase,
    statusBefore: 'active' as GameStatus,
    statusAfter: 'active' as GameStatus,
    progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 } as ProgressSnapshot,
    progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 } as ProgressSnapshot,
    ...overrides,
  };
}

// Helper to create a victory result
function createVictoryResult(overrides: Partial<GameResult> = {}): GameResult {
  return {
    winner: 1,
    reason: 'ring_elimination',
    finalScore: {
      ringsEliminated: { 1: 0, 2: 18 },
      territorySpaces: { 1: 10, 2: 5 },
      ringsRemaining: { 1: 18, 2: 0 },
    },
    ...overrides,
  };
}

describe('GameEventLog', () => {
  describe('basic rendering', () => {
    it('renders the component with data-testid', () => {
      render(<GameEventLog history={[]} />);
      expect(screen.getByTestId('game-event-log')).toBeInTheDocument();
    });

    it('displays "Game log" header', () => {
      render(<GameEventLog history={[]} />);
      expect(screen.getByText('Game log')).toBeInTheDocument();
    });

    it('displays "No events yet." when history is empty', () => {
      render(<GameEventLog history={[]} />);
      expect(screen.getByText('No events yet.')).toBeInTheDocument();
    });

    it('has scrollable container with max height', () => {
      render(<GameEventLog history={[]} />);
      const container = screen.getByTestId('game-event-log');
      expect(container).toHaveClass('max-h-64', 'overflow-y-auto');
    });
  });

  describe('legacy props interface', () => {
    it('renders move entries from history', () => {
      const history = [createHistoryEntry({ moveNumber: 1 })];
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P1 placed/)).toBeInTheDocument();
    });

    it('renders multiple move entries in reverse order (most recent first)', () => {
      const history = [
        createHistoryEntry({
          moveNumber: 1,
          action: createMove({ type: 'place_ring', player: 1, moveNumber: 1 }),
        }),
        createHistoryEntry({
          moveNumber: 2,
          action: createMove({ type: 'place_ring', player: 2, moveNumber: 2 }),
        }),
      ];
      render(<GameEventLog history={history} />);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveTextContent('P2 placed');
      expect(items[1]).toHaveTextContent('P1 placed');
    });

    it('renders system events', () => {
      render(<GameEventLog history={[]} systemEvents={['Game started', 'Player connected']} />);
      expect(screen.getByText('Game started')).toBeInTheDocument();
      expect(screen.getByText('Player connected')).toBeInTheDocument();
    });

    it('displays "System events" section label when system events exist', () => {
      render(<GameEventLog history={[]} systemEvents={['Test event']} />);
      expect(screen.getByText('System events')).toBeInTheDocument();
    });

    it('respects maxEntries limit', () => {
      const history = Array.from({ length: 100 }, (_, i) =>
        createHistoryEntry({
          moveNumber: i + 1,
          action: createMove({ moveNumber: i + 1 }),
        })
      );
      render(<GameEventLog history={history} maxEntries={5} />);
      const items = screen.getAllByRole('listitem');
      // Only recent moves from maxEntries limit should appear
      expect(items.length).toBeLessThanOrEqual(10); // some buffer for system events
    });
  });

  describe('victory display', () => {
    it('displays victory message when victoryState is provided', () => {
      const victory = createVictoryResult({ winner: 1, reason: 'ring_elimination' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText(/Player P1 wins by Ring Elimination/)).toBeInTheDocument();
    });

    it('displays draw message when victory reason is draw', () => {
      const victory = createVictoryResult({ winner: undefined, reason: 'draw' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText('Game ended in a draw.')).toBeInTheDocument();
    });

    it('displays victory by territory control', () => {
      const victory = createVictoryResult({ winner: 2, reason: 'territory_control' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText(/Player P2 wins by Territory Control/)).toBeInTheDocument();
    });

    it('displays victory by last player standing', () => {
      const victory = createVictoryResult({ winner: 1, reason: 'last_player_standing' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText(/Player P1 wins by Last Player Standing/)).toBeInTheDocument();
    });

    it('displays victory by timeout', () => {
      const victory = createVictoryResult({ winner: 1, reason: 'timeout' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText(/Player P1 wins by Timeout/)).toBeInTheDocument();
    });

    it('displays victory by resignation', () => {
      const victory = createVictoryResult({ winner: 1, reason: 'resignation' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText(/Player P1 wins by Resignation/)).toBeInTheDocument();
    });

    it('displays abandonment result', () => {
      const victory = createVictoryResult({ winner: undefined, reason: 'abandonment' });
      render(<GameEventLog history={[]} victoryState={victory} />);
      expect(screen.getByText(/Abandonment/)).toBeInTheDocument();
    });

    it('applies victory styling to victory entry', () => {
      const victory = createVictoryResult();
      render(<GameEventLog history={[]} victoryState={victory} />);
      const victoryElement = screen.getByText(/Player P1 wins/);
      expect(victoryElement.closest('div')).toHaveClass('bg-emerald-900/40');
    });
  });

  describe('move type formatting', () => {
    it('formats swap_sides moves with a pie rule description', () => {
      const history = [
        createHistoryEntry({
          moveNumber: 3,
          action: createMove({
            type: 'swap_sides',
            player: 2,
            moveNumber: 3,
          }),
        }),
      ];

      render(<GameEventLog history={history} />);
      expect(
        screen.getByText('#3 — P2 invoked the pie rule and swapped colours with P1')
      ).toBeInTheDocument();
    });

    it('formats place_ring moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'place_ring',
            player: 1,
            to: { x: 3, y: 4 },
            placementCount: 1,
          }),
        }),
      ];
      // Default boardType is square8 with squareRankFromBottom: d4 = (x:3, y:4)
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P1 placed 1 ring at d4/)).toBeInTheDocument();
    });

    it('formats multiple ring placements correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            placementCount: 3,
          }),
        }),
      ];
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P1 placed 3 rings/)).toBeInTheDocument();
    });

    it('formats move_stack moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'move_stack',
            player: 2,
            from: { x: 0, y: 0 },
            to: { x: 3, y: 3 },
          }),
        }),
      ];
      // With squareRankFromBottom (default for square8): a8 = (0,0), d5 = (3,3)
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P2 moved from a8 to d5/)).toBeInTheDocument();
    });

    it('formats overtaking_capture moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'overtaking_capture',
            player: 1,
            from: { x: 0, y: 0 },
            captureTarget: { x: 1, y: 1 },
            to: { x: 2, y: 2 },
            overtakenRings: [2, 2],
          }),
        }),
      ];
      // With squareRankFromBottom: a8 = (0,0), b7 = (1,1), c6 = (2,2)
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P1 capture from a8 over b7 to c6 x2/)).toBeInTheDocument();
    });

    it('formats continue_capture_segment moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'continue_capture_segment',
            player: 1,
            from: { x: 2, y: 2 },
            captureTarget: { x: 3, y: 3 },
            to: { x: 4, y: 4 },
            overtakenRings: [2],
          }),
        }),
      ];
      // With squareRankFromBottom: d5 = (3,3), e4 = (4,4)
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P1 continued capture over d5 to e4 x1/)).toBeInTheDocument();
    });

    it('formats skip_placement moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({ type: 'skip_placement', player: 1 }),
        }),
      ];
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/P1 skipped placement/)).toBeInTheDocument();
    });

    it('formats build_stack moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'build_stack',
            player: 1,
            from: { x: 2, y: 2 },
            to: { x: 3, y: 3 },
            buildAmount: 2,
          }),
        }),
      ];
      // With squareRankFromBottom: d5 = (3,3)
      render(<GameEventLog history={history} />);
      expect(screen.getByText(/#1 — P1 built stack at d5/)).toBeInTheDocument();
    });

    it('formats process_line moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'process_line',
            player: 1,
            formedLines: [{ positions: [], player: 1, length: 3, direction: { x: 1, y: 0 } }],
          }),
        }),
      ];
      render(<GameEventLog history={history} />);
      // The format includes move number and [Line order] prefix, e.g. "#1 — P1 [Line order] processed 1 line"
      expect(screen.getByText(/\[Line order\] processed 1 line/)).toBeInTheDocument();
    });

    it('formats territory processing moves correctly', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'process_territory_region',
            player: 1,
            claimedTerritory: [{ spaces: [], controllingPlayer: 1, isDisconnected: false }],
            eliminatedRings: [{ player: 2, count: 3 }],
          }),
        }),
      ];
      render(<GameEventLog history={history} />);
      expect(
        screen.getByText(
          /Territory \/ elimination processing by P1 \(1 region, 3 rings eliminated\)/
        )
      ).toBeInTheDocument();
    });
  });

  describe('hexagonal position formatting', () => {
    it('displays hex positions in algebraic notation for hex boards', () => {
      const history = [
        createHistoryEntry({
          action: createMove({
            type: 'move_stack',
            player: 1,
            // Hex coordinates within hex8 board (radius 4, size 9)
            from: { x: 0, y: 0, z: 0 },
            to: { x: 1, y: -1, z: 0 },
          }),
        }),
      ];
      // Pass boardType: 'hex8' to enable hex coordinate formatting
      render(<GameEventLog history={history} boardType="hex8" />);
      // Hex boards use algebraic-like notation (file+rank mapped from q,r)
      // For hex8 (radius 4): q=0 -> rank 5, r=0 -> file 'e'
      expect(screen.getByText(/e5/)).toBeInTheDocument();
    });
  });

  describe('view model props interface', () => {
    it('renders using view model when provided', () => {
      const viewModel: EventLogViewModel = {
        entries: [{ key: 'move-1', text: 'Custom move text', type: 'move', moveNumber: 1 }],
        hasContent: true,
      };
      render(<GameEventLog viewModel={viewModel} />);
      expect(screen.getByText('Custom move text')).toBeInTheDocument();
    });

    it('displays victory message from view model', () => {
      const viewModel: EventLogViewModel = {
        entries: [{ key: 'victory', text: 'Player 1 wins by Ring Elimination', type: 'victory' }],
        victoryMessage: 'Player 1 wins by Ring Elimination',
        hasContent: true,
      };
      render(<GameEventLog viewModel={viewModel} />);
      expect(screen.getByText('Player 1 wins by Ring Elimination')).toBeInTheDocument();
    });

    it('displays system events from view model', () => {
      const viewModel: EventLogViewModel = {
        entries: [{ key: 'system-0', text: 'System event from view model', type: 'system' }],
        hasContent: true,
      };
      render(<GameEventLog viewModel={viewModel} />);
      expect(screen.getByText('System event from view model')).toBeInTheDocument();
    });

    it('shows empty state when view model has no content', () => {
      const viewModel: EventLogViewModel = {
        entries: [],
        hasContent: false,
      };
      render(<GameEventLog viewModel={viewModel} />);
      expect(screen.getByText('No events yet.')).toBeInTheDocument();
    });
  });

  describe('section labels', () => {
    it('displays "Recent moves" label when moves exist', () => {
      const history = [createHistoryEntry()];
      render(<GameEventLog history={history} />);
      expect(screen.getByText('Recent moves')).toBeInTheDocument();
    });

    it('does not display "Recent moves" label when no moves exist', () => {
      render(<GameEventLog history={[]} systemEvents={['System event']} />);
      expect(screen.queryByText('Recent moves')).not.toBeInTheDocument();
    });
  });

  describe('styling and layout', () => {
    it('applies correct border and background styling', () => {
      render(<GameEventLog history={[]} />);
      const container = screen.getByTestId('game-event-log');
      // Updated to match current component styling (bg-slate-900/70, with rounded-xl)
      expect(container).toHaveClass('border', 'border-slate-700', 'bg-slate-900/70', 'rounded-xl');
    });

    it('renders moves in a list', () => {
      const history = [createHistoryEntry(), createHistoryEntry({ moveNumber: 2 })];
      render(<GameEventLog history={history} />);
      const listItems = screen.getAllByRole('listitem');
      expect(listItems.length).toBeGreaterThan(0);
    });
  });
});
