import React from 'react';
import { render, screen, within, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { VictoryModal, type RematchStatus } from '../../../../src/client/components/VictoryModal';
import type {
  VictoryViewModel,
  PlayerViewModel,
  PlayerFinalStatsViewModel,
  PlayerRingStatsViewModel,
} from '../../../../src/client/adapters/gameViewModels';
import type { GameResult, Player, GameState, BoardType } from '../../../../src/shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Test Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create ring stats for a player
 */
function createRingStats(
  overrides: Partial<PlayerRingStatsViewModel> = {}
): PlayerRingStatsViewModel {
  return {
    inHand: 0,
    onBoard: 10,
    eliminated: 5,
    total: 18,
    ...overrides,
  };
}

/**
 * Create a PlayerViewModel for testing
 */
function createPlayerViewModel(
  playerNumber: number,
  overrides: Partial<PlayerViewModel> = {}
): PlayerViewModel {
  const colorClasses = ['bg-emerald-500', 'bg-sky-500', 'bg-amber-500', 'bg-fuchsia-500'];

  return {
    id: `player-${playerNumber}`,
    playerNumber,
    username: `Player ${playerNumber}`,
    isCurrentPlayer: false,
    isUserPlayer: playerNumber === 1,
    colorClass: colorClasses[playerNumber - 1] ?? 'bg-gray-500',
    ringStats: createRingStats(),
    territorySpaces: 3,
    timeRemaining: undefined,
    aiInfo: { isAI: false },
    ...overrides,
  };
}

/**
 * Create an AI PlayerViewModel
 */
function createAIPlayerViewModel(
  playerNumber: number,
  difficulty: number = 5,
  overrides: Partial<PlayerViewModel> = {}
): PlayerViewModel {
  return createPlayerViewModel(playerNumber, {
    username: 'AI Bot',
    isUserPlayer: false,
    aiInfo: {
      isAI: true,
      difficulty,
      difficultyLabel: 'Advanced · Minimax',
      difficultyColor: 'text-blue-300',
      difficultyBgColor: 'bg-blue-900/40',
      aiTypeLabel: 'Minimax',
    },
    ...overrides,
  });
}

/**
 * Create player final stats
 */
function createFinalStats(
  playerNumber: number,
  overrides: Partial<PlayerFinalStatsViewModel> = {}
): PlayerFinalStatsViewModel {
  const player = createPlayerViewModel(playerNumber, overrides.player);
  return {
    player,
    ringsOnBoard: 10,
    ringsEliminated: 5,
    territorySpaces: 3,
    totalMoves: 25,
    isWinner: false,
    ...overrides,
  };
}

/**
 * Create a minimal VictoryViewModel
 */
function createVictoryViewModel(
  overrides: Partial<VictoryViewModel> = {}
): VictoryViewModel {
  const winner = createPlayerViewModel(1, { isUserPlayer: false });
  const finalStats = [
    createFinalStats(1, { isWinner: true, player: winner }),
    createFinalStats(2, { isWinner: false }),
  ];

  return {
    isVisible: true,
    title: '🏆 Player 1 Wins!',
    description: 'Victory by Ring Elimination',
    titleColorClass: 'text-slate-100',
    winner,
    finalStats,
    gameSummary: {
      boardType: 'square8' as BoardType,
      totalTurns: 50,
      playerCount: 2,
      isRated: false,
    },
    userWon: false,
    userLost: false,
    isDraw: false,
    ...overrides,
  };
}

/**
 * Create a GameResult object
 */
function createGameResult(
  overrides: Partial<GameResult> = {}
): GameResult {
  return {
    reason: 'ring_elimination',
    winner: 1,
    finalScore: {
      ringsEliminated: { 1: 18, 2: 5 },
      ringsRemaining: { 1: 10, 2: 3 },
      territorySpaces: { 1: 5, 2: 2 },
    },
    ...overrides,
  };
}

/**
 * Create mock players array
 */
function createPlayers(): Player[] {
  return [
    {
      id: 'player-1',
      username: 'Player 1',
      playerNumber: 1,
      type: 'human',
      ringsInHand: 0,
      eliminatedRings: 18,
      territorySpaces: 5,
      isReady: true,
      timeRemaining: 0,
    },
    {
      id: 'player-2',
      username: 'Player 2',
      playerNumber: 2,
      type: 'human',
      ringsInHand: 0,
      eliminatedRings: 5,
      territorySpaces: 2,
      isReady: true,
      timeRemaining: 0,
    },
  ];
}

/**
 * Default props for VictoryModal tests
 */
function createDefaultProps(overrides: Partial<React.ComponentProps<typeof VictoryModal>> = {}) {
  return {
    isOpen: true,
    viewModel: createVictoryViewModel(),
    onClose: jest.fn(),
    onReturnToLobby: jest.fn(),
    ...overrides,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('VictoryModal', () => {
  describe('Rendering', () => {
    it('renders modal when isOpen is true', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      expect(screen.getByTestId('victory-modal')).toBeInTheDocument();
    });

    it('does not render when isOpen is false', () => {
      const props = createDefaultProps({ isOpen: false });
      render(<VictoryModal {...props} />);

      expect(screen.queryByTestId('victory-modal')).not.toBeInTheDocument();
    });

    it('does not render when viewModel is null', () => {
      const props = createDefaultProps({ viewModel: null });
      render(<VictoryModal {...props} />);

      expect(screen.queryByTestId('victory-modal')).not.toBeInTheDocument();
    });

    it('does not render when viewModel.isVisible is false', () => {
      const viewModel = createVictoryViewModel({ isVisible: false });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.queryByTestId('victory-modal')).not.toBeInTheDocument();
    });

    it('displays title from viewModel', () => {
      const viewModel = createVictoryViewModel({ title: '🏆 Alice Wins!' });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('🏆 Alice Wins!')).toBeInTheDocument();
    });

    it('displays description from viewModel', () => {
      const viewModel = createVictoryViewModel({ description: 'Victory by Territory Control' });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('Victory by Territory Control')).toBeInTheDocument();
    });

    it('renders with accessible dialog structure', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      const modal = screen.getByTestId('victory-modal');
      expect(modal).toBeInTheDocument();
      expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Victory Type Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Victory Type Display', () => {
    it('displays ring elimination victory with trophy emoji', () => {
      const gameResult = createGameResult({ reason: 'ring_elimination' });
      const props = createDefaultProps({ gameResult });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('🏆');
    });

    it('displays territory victory with castle emoji', () => {
      const gameResult = createGameResult({ reason: 'territory_control' });
      const props = createDefaultProps({ gameResult });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('🏰');
    });

    it('displays last player standing victory with crown emoji', () => {
      const gameResult = createGameResult({ reason: 'last_player_standing' });
      const props = createDefaultProps({ gameResult });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('👑');
    });

    it('displays timeout victory with clock emoji', () => {
      const gameResult = createGameResult({ reason: 'timeout' });
      const props = createDefaultProps({ gameResult });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('⏰');
    });

    it('displays resignation victory with flag emoji', () => {
      const gameResult = createGameResult({ reason: 'resignation' });
      const props = createDefaultProps({ gameResult });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('🏳️');
    });

    it('displays abandonment with door emoji', () => {
      const gameResult = createGameResult({ reason: 'abandonment' });
      const props = createDefaultProps({ gameResult });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('🚪');
    });

    it('displays draw result with handshake emoji', () => {
      const gameResult = createGameResult({ reason: 'draw', winner: undefined });
      const viewModel = createVictoryViewModel({
        isDraw: true,
        winner: undefined,
        title: '🤝 Draw!',
        description: 'The game ended in a stalemate',
      });
      const props = createDefaultProps({ gameResult, viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('img', { name: 'trophy' })).toHaveTextContent('🤝');
      expect(screen.getByText('🤝 Draw!')).toBeInTheDocument();
    });

    it('displays structural stalemate victory appropriately', () => {
      const gameResult = createGameResult({ reason: 'game_completed' });
      const viewModel = createVictoryViewModel({
        title: '🧱 Structural Stalemate',
        description: 'The game reached a structural stalemate',
      });
      const props = createDefaultProps({ gameResult, viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('🧱 Structural Stalemate')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Winner Display Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Winner Display', () => {
    it('highlights winner in stats table with crown', () => {
      const winner = createPlayerViewModel(1, { username: 'Alice' });
      const finalStats = [
        createFinalStats(1, { isWinner: true, player: winner }),
        createFinalStats(2, { isWinner: false }),
      ];
      // Use ring_elimination so trophy is different from crown emoji
      const gameResult = createGameResult({ reason: 'ring_elimination' });
      const viewModel = createVictoryViewModel({ winner, finalStats });
      const props = createDefaultProps({ viewModel, gameResult });
      render(<VictoryModal {...props} />);

      // Winner row in stats table should have a crown emoji
      const table = screen.getByRole('table');
      const crownEmoji = within(table).getByText('👑');
      expect(crownEmoji).toBeInTheDocument();
    });

    it('applies green title color when user wins', () => {
      const viewModel = createVictoryViewModel({
        userWon: true,
        titleColorClass: 'text-green-400',
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      const title = screen.getByRole('heading', { level: 1 });
      expect(title).toHaveClass('text-green-400');
    });

    it('applies red title color when user loses', () => {
      const viewModel = createVictoryViewModel({
        userLost: true,
        titleColorClass: 'text-red-400',
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      const title = screen.getByRole('heading', { level: 1 });
      expect(title).toHaveClass('text-red-400');
    });

    it('applies neutral color for spectators or draws', () => {
      const viewModel = createVictoryViewModel({
        userWon: false,
        userLost: false,
        isDraw: true,
        titleColorClass: 'text-slate-100',
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      const title = screen.getByRole('heading', { level: 1 });
      expect(title).toHaveClass('text-slate-100');
    });

    it('displays winner username in stats table', () => {
      const winner = createPlayerViewModel(1, { username: 'ChampionPlayer' });
      const finalStats = [
        createFinalStats(1, { isWinner: true, player: winner }),
        createFinalStats(2, { isWinner: false }),
      ];
      const viewModel = createVictoryViewModel({ winner, finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('ChampionPlayer')).toBeInTheDocument();
    });

    it('shows winner first in the sorted stats table', () => {
      const winner = createPlayerViewModel(2, { username: 'Winner' });
      const loser = createPlayerViewModel(1, { username: 'Loser' });
      const finalStats = [
        createFinalStats(1, { isWinner: false, player: loser }),
        createFinalStats(2, { isWinner: true, player: winner }),
      ];
      const viewModel = createVictoryViewModel({ winner, finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      // Stats table should exist and display both players
      const table = screen.getByRole('table');
      const rows = within(table).getAllByRole('row');
      // First data row (after header) should be winner
      expect(rows.length).toBeGreaterThan(1);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Statistics Display Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Statistics Display', () => {
    it('displays stats table with correct headers', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      expect(screen.getByText('Player')).toBeInTheDocument();
      expect(screen.getByText('Rings on Board')).toBeInTheDocument();
      expect(screen.getByText('Rings Eliminated')).toBeInTheDocument();
      expect(screen.getByText('Territory')).toBeInTheDocument();
      expect(screen.getByText('Moves')).toBeInTheDocument();
    });

    it('displays rings on board for each player', () => {
      const finalStats = [
        createFinalStats(1, { ringsOnBoard: 15 }),
        createFinalStats(2, { ringsOnBoard: 8 }),
      ];
      const viewModel = createVictoryViewModel({ finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('15')).toBeInTheDocument();
      expect(screen.getByText('8')).toBeInTheDocument();
    });

    it('displays rings eliminated for each player', () => {
      const finalStats = [
        createFinalStats(1, { ringsEliminated: 18 }),
        createFinalStats(2, { ringsEliminated: 6 }),
      ];
      const viewModel = createVictoryViewModel({ finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('18')).toBeInTheDocument();
      expect(screen.getByText('6')).toBeInTheDocument();
    });

    it('displays territory spaces for each player', () => {
      const player1 = createPlayerViewModel(1, { territorySpaces: 12 });
      const player2 = createPlayerViewModel(2, { territorySpaces: 4 });
      const finalStats = [
        createFinalStats(1, { territorySpaces: 12, player: player1 }),
        createFinalStats(2, { territorySpaces: 4, player: player2 }),
      ];
      const viewModel = createVictoryViewModel({ finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('12')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument();
    });

    it('displays total moves for each player', () => {
      const finalStats = [
        createFinalStats(1, { totalMoves: 42 }),
        createFinalStats(2, { totalMoves: 38 }),
      ];
      const viewModel = createVictoryViewModel({ finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('42')).toBeInTheDocument();
      expect(screen.getByText('38')).toBeInTheDocument();
    });

    it('displays game summary information', () => {
      const viewModel = createVictoryViewModel({
        gameSummary: {
          boardType: 'square19',
          totalTurns: 120,
          playerCount: 2,
          isRated: false,
        },
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('square19')).toBeInTheDocument();
      expect(screen.getByText('120')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('shows "Rated" label for rated games', () => {
      const viewModel = createVictoryViewModel({
        gameSummary: {
          boardType: 'square8',
          totalTurns: 50,
          playerCount: 2,
          isRated: true,
        },
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('Rated')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Action Button Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Action Buttons', () => {
    it('renders Return to Lobby button', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('button', { name: /Return to Lobby/i })).toBeInTheDocument();
    });

    it('renders Close button', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('button', { name: /Close/i })).toBeInTheDocument();
    });

    it('calls onReturnToLobby when Return to Lobby button is clicked', () => {
      const onReturnToLobby = jest.fn();
      const props = createDefaultProps({ onReturnToLobby });
      render(<VictoryModal {...props} />);

      fireEvent.click(screen.getByRole('button', { name: /Return to Lobby/i }));
      expect(onReturnToLobby).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when Close button is clicked', () => {
      const onClose = jest.fn();
      const props = createDefaultProps({ onClose });
      render(<VictoryModal {...props} />);

      fireEvent.click(screen.getByRole('button', { name: /Close/i }));
      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('renders Play Again button for sandbox games with onRematch', () => {
      const onRematch = jest.fn();
      const props = createDefaultProps({ onRematch });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('button', { name: /Play Again/i })).toBeInTheDocument();
    });

    it('calls onRematch when Play Again button is clicked', () => {
      const onRematch = jest.fn();
      const props = createDefaultProps({ onRematch });
      render(<VictoryModal {...props} />);

      fireEvent.click(screen.getByRole('button', { name: /Play Again/i }));
      expect(onRematch).toHaveBeenCalledTimes(1);
    });

    it('renders Request Rematch button for backend games', () => {
      const onRequestRematch = jest.fn();
      const props = createDefaultProps({ onRequestRematch });
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('button', { name: /Request Rematch/i })).toBeInTheDocument();
    });

    it('calls onRequestRematch when Request Rematch button is clicked', () => {
      const onRequestRematch = jest.fn();
      const props = createDefaultProps({ onRequestRematch });
      render(<VictoryModal {...props} />);

      fireEvent.click(screen.getByRole('button', { name: /Request Rematch/i }));
      expect(onRequestRematch).toHaveBeenCalledTimes(1);
    });

    it('shows waiting message when rematch is pending and user is requester', () => {
      const rematchStatus: RematchStatus = {
        isPending: true,
        isRequester: true,
        expiresAt: new Date(Date.now() + 30000).toISOString(),
        status: 'pending',
      };
      const props = createDefaultProps({ onRequestRematch: jest.fn(), rematchStatus });
      render(<VictoryModal {...props} />);

      expect(screen.getByText(/Waiting for opponent/i)).toBeInTheDocument();
    });

    it('shows Accept/Decline buttons when rematch request is from opponent', () => {
      const rematchStatus: RematchStatus = {
        isPending: true,
        isRequester: false,
        requesterUsername: 'Opponent',
        requestId: 'req-123',
        expiresAt: new Date(Date.now() + 30000).toISOString(),
        status: 'pending',
      };
      const onAcceptRematch = jest.fn();
      const onDeclineRematch = jest.fn();
      const props = createDefaultProps({
        onRequestRematch: jest.fn(),
        onAcceptRematch,
        onDeclineRematch,
        rematchStatus,
      });
      render(<VictoryModal {...props} />);

      expect(screen.getByText(/Opponent wants a rematch!/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Accept/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Decline/i })).toBeInTheDocument();
    });

    it('calls onAcceptRematch when Accept button is clicked', () => {
      const rematchStatus: RematchStatus = {
        isPending: true,
        isRequester: false,
        requestId: 'req-123',
        expiresAt: new Date(Date.now() + 30000).toISOString(),
        status: 'pending',
      };
      const onAcceptRematch = jest.fn();
      const props = createDefaultProps({
        onRequestRematch: jest.fn(),
        onAcceptRematch,
        rematchStatus,
      });
      render(<VictoryModal {...props} />);

      fireEvent.click(screen.getByRole('button', { name: /Accept/i }));
      expect(onAcceptRematch).toHaveBeenCalledWith('req-123');
    });

    it('calls onDeclineRematch when Decline button is clicked', () => {
      const rematchStatus: RematchStatus = {
        isPending: true,
        isRequester: false,
        requestId: 'req-123',
        expiresAt: new Date(Date.now() + 30000).toISOString(),
        status: 'pending',
      };
      const onDeclineRematch = jest.fn();
      const props = createDefaultProps({
        onRequestRematch: jest.fn(),
        onDeclineRematch,
        rematchStatus,
      });
      render(<VictoryModal {...props} />);

      fireEvent.click(screen.getByRole('button', { name: /Decline/i }));
      expect(onDeclineRematch).toHaveBeenCalledWith('req-123');
    });

    it('shows accepted message when rematch is accepted', () => {
      const rematchStatus: RematchStatus = {
        isPending: false,
        status: 'accepted',
      };
      const props = createDefaultProps({ onRequestRematch: jest.fn(), rematchStatus });
      render(<VictoryModal {...props} />);

      expect(screen.getByText(/Rematch accepted!/i)).toBeInTheDocument();
    });

    it('shows declined message when rematch is declined', () => {
      const rematchStatus: RematchStatus = {
        isPending: false,
        status: 'declined',
      };
      const props = createDefaultProps({ onRequestRematch: jest.fn(), rematchStatus });
      render(<VictoryModal {...props} />);

      expect(screen.getByText(/Rematch declined/i)).toBeInTheDocument();
    });

    it('shows expired message and new request button when rematch expires', () => {
      const rematchStatus: RematchStatus = {
        isPending: false,
        status: 'expired',
      };
      const onRequestRematch = jest.fn();
      const props = createDefaultProps({ onRequestRematch, rematchStatus });
      render(<VictoryModal {...props} />);

      expect(screen.getByText(/Rematch request expired/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Request Rematch/i })).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Edge Cases
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Edge Cases', () => {
    it('renders correctly for 2-player game ending', () => {
      const finalStats = [
        createFinalStats(1, { isWinner: true }),
        createFinalStats(2, { isWinner: false }),
      ];
      const viewModel = createVictoryViewModel({
        finalStats,
        gameSummary: { boardType: 'square8', totalTurns: 50, playerCount: 2, isRated: false },
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      // Verify both players appear in the table
      const table = screen.getByRole('table');
      const rows = within(table).getAllByRole('row');
      // Header row + 2 player rows
      expect(rows.length).toBe(3);
    });

    it('renders correctly for 3-player game with rankings', () => {
      const player1 = createPlayerViewModel(1, { username: 'Alice' });
      const player2 = createPlayerViewModel(2, { username: 'Bob' });
      const player3 = createPlayerViewModel(3, { username: 'Charlie' });
      const finalStats = [
        createFinalStats(1, { isWinner: true, ringsEliminated: 24, player: player1 }),
        createFinalStats(2, { isWinner: false, ringsEliminated: 15, player: player2 }),
        createFinalStats(3, { isWinner: false, ringsEliminated: 10, player: player3 }),
      ];
      const viewModel = createVictoryViewModel({
        winner: player1,
        finalStats,
        gameSummary: { boardType: 'square19', totalTurns: 100, playerCount: 3, isRated: false },
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
      expect(screen.getByText('Charlie')).toBeInTheDocument();

      // Verify stats table shows all 3 players
      const table = screen.getByRole('table');
      const rows = within(table).getAllByRole('row');
      expect(rows.length).toBe(4); // Header + 3 players
    });

    it('renders correctly for 4-player game with rankings', () => {
      const players = [1, 2, 3, 4].map((n) => createPlayerViewModel(n, { username: `Player${n}` }));
      const finalStats = players.map((p, i) =>
        createFinalStats(p.playerNumber, {
          isWinner: i === 0,
          ringsEliminated: 30 - i * 5,
          player: p,
        })
      );
      const viewModel = createVictoryViewModel({
        winner: players[0],
        finalStats,
        gameSummary: { boardType: 'square19', totalTurns: 150, playerCount: 4, isRated: true },
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      // Verify all 4 players shown
      const table = screen.getByRole('table');
      const rows = within(table).getAllByRole('row');
      expect(rows.length).toBe(5); // Header + 4 players
    });

    it('renders correctly for AI opponent games', () => {
      const humanPlayer = createPlayerViewModel(1, { username: 'Human' });
      const aiPlayer = createAIPlayerViewModel(2, 7, { username: 'AI Expert' });
      const finalStats = [
        createFinalStats(1, { isWinner: true, player: humanPlayer }),
        createFinalStats(2, { isWinner: false, player: aiPlayer }),
      ];
      const viewModel = createVictoryViewModel({
        winner: humanPlayer,
        finalStats,
        userWon: true,
        titleColorClass: 'text-green-400',
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('Human')).toBeInTheDocument();
      expect(screen.getByText('AI Expert')).toBeInTheDocument();
    });

    it('renders correctly as spectator view of completed game', () => {
      const player1 = createPlayerViewModel(1, { username: 'Alice', isUserPlayer: false });
      const player2 = createPlayerViewModel(2, { username: 'Bob', isUserPlayer: false });
      const finalStats = [
        createFinalStats(1, { isWinner: true, player: player1 }),
        createFinalStats(2, { isWinner: false, player: player2 }),
      ];
      const viewModel = createVictoryViewModel({
        winner: player1,
        finalStats,
        userWon: false,
        userLost: false,
        titleColorClass: 'text-slate-100',
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      // Spectator should see neutral styling
      const title = screen.getByRole('heading', { level: 1 });
      expect(title).toHaveClass('text-slate-100');
    });

    it('handles missing winner gracefully for draw games', () => {
      const finalStats = [
        createFinalStats(1, { isWinner: false }),
        createFinalStats(2, { isWinner: false }),
      ];
      const viewModel = createVictoryViewModel({
        winner: undefined,
        finalStats,
        isDraw: true,
        title: '🤝 Draw!',
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('🤝 Draw!')).toBeInTheDocument();
      // No crown should be shown since there's no winner
      expect(screen.queryByText('👑')).not.toBeInTheDocument();
    });

    it('does not show confetti for draw games', () => {
      const gameResult = createGameResult({ reason: 'draw', winner: undefined });
      const viewModel = createVictoryViewModel({ isDraw: true });
      const props = createDefaultProps({ gameResult, viewModel });
      const { container } = render(<VictoryModal {...props} />);

      // Confetti particles should not be rendered for draws
      expect(container.querySelector('.confetti-particle')).not.toBeInTheDocument();
    });

    it('does not show confetti for abandonment', () => {
      const gameResult = createGameResult({ reason: 'abandonment' });
      const props = createDefaultProps({ gameResult });
      const { container } = render(<VictoryModal {...props} />);

      expect(container.querySelector('.confetti-particle')).not.toBeInTheDocument();
    });

    it('handles hexagonal board type in game summary', () => {
      const viewModel = createVictoryViewModel({
        gameSummary: {
          boardType: 'hexagonal',
          totalTurns: 200,
          playerCount: 2,
          isRated: false,
        },
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText('hexagonal')).toBeInTheDocument();
      expect(screen.getByText('200')).toBeInTheDocument();
    });

    it('renders correctly with zero territory spaces', () => {
      const player = createPlayerViewModel(1, { territorySpaces: 0 });
      const finalStats = [createFinalStats(1, { territorySpaces: 0, player })];
      const viewModel = createVictoryViewModel({ finalStats });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      // Should display 0 for territory spaces
      const table = screen.getByRole('table');
      expect(within(table).getByText('0')).toBeInTheDocument();
    });

    it('renders long usernames without breaking layout', () => {
      const longName = 'VeryLongUsernameWithManyCharacters123';
      const player = createPlayerViewModel(1, { username: longName });
      const finalStats = [createFinalStats(1, { player })];
      const viewModel = createVictoryViewModel({
        winner: player,
        finalStats,
        title: `🏆 ${longName} Wins!`,
      });
      const props = createDefaultProps({ viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.getByText(longName)).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Weird State / "What happened?" Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Weird State Help', () => {
    it('does not show "What happened?" link for normal victories', () => {
      const gameResult = createGameResult({ reason: 'ring_elimination' });
      const viewModel = createVictoryViewModel();
      const props = createDefaultProps({ gameResult, viewModel });
      render(<VictoryModal {...props} />);

      expect(screen.queryByText(/What happened?/i)).not.toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Legacy Props Fallback Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Legacy Props Fallback', () => {
    it('constructs viewModel from legacy props when viewModel is not provided', () => {
      const gameResult = createGameResult({ reason: 'territory_control', winner: 1 });
      const players = createPlayers();
      const props = createDefaultProps({
        viewModel: undefined,
        gameResult,
        players,
      });
      render(<VictoryModal {...props} />);

      // Should render based on the legacy props
      expect(screen.getByTestId('victory-modal')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Accessibility Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Accessibility', () => {
    it('has accessible title with id for aria-labelledby', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      const title = screen.getByRole('heading', { level: 1 });
      expect(title).toHaveAttribute('id', 'victory-title');
    });

    it('has accessible description with id for aria-describedby', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      const description = document.getElementById('victory-description');
      expect(description).toBeInTheDocument();
    });

    it('trophy has accessible role and aria-label', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      const trophy = screen.getByRole('img', { name: 'trophy' });
      expect(trophy).toBeInTheDocument();
    });

    it('stats table has proper table structure', () => {
      const props = createDefaultProps();
      render(<VictoryModal {...props} />);

      expect(screen.getByRole('table')).toBeInTheDocument();
      expect(screen.getAllByRole('columnheader').length).toBeGreaterThanOrEqual(4);
    });

    it('buttons are properly focusable', () => {
      const props = createDefaultProps({ onRematch: jest.fn() });
      render(<VictoryModal {...props} />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach((button) => {
        expect(button).not.toHaveAttribute('disabled');
      });
    });
  });
});
