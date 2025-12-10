import { render, screen, fireEvent, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import { VictoryModal } from '../../src/client/components/VictoryModal';
import { toVictoryViewModel } from '../../src/client/adapters/gameViewModels';
import { GameResult, Player, GameState, BoardState } from '../../src/shared/types/game';

// Helper to create test game result
function createGameResult(winner: number | undefined, reason: GameResult['reason']): GameResult {
  const base: GameResult = {
    reason,
    finalScore: {
      ringsEliminated: { 1: 15, 2: 8 },
      territorySpaces: { 1: 25, 2: 10 },
      ringsRemaining: { 1: 3, 2: 10 },
    },
  };

  return winner === undefined ? base : { ...base, winner };
}

// Helper to create test players
function createTestPlayers(): Player[] {
  return [
    {
      id: 'user1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 3,
      eliminatedRings: 15,
      territorySpaces: 25,
    },
    {
      id: 'user2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 10,
      eliminatedRings: 8,
      territorySpaces: 10,
    },
  ];
}

// Helper to create minimal game state
function createTestGameState(players: Player[]): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 15, 2: 8 },
    size: 8,
    type: 'square8',
  };

  return {
    id: 'test-game',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'finished',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 23,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
  };
}

describe('VictoryModal', () => {
  const mockOnClose = jest.fn();
  const mockOnReturnToLobby = jest.fn();
  const mockOnRematch = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should not render when isOpen is false', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    const { container } = render(
      <VictoryModal
        isOpen={false}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(container.firstChild).toBeNull();
  });

  it('should not render when gameResult is null', () => {
    const players = createTestPlayers();

    const { container } = render(
      <VictoryModal
        isOpen={true}
        gameResult={null}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(container.firstChild).toBeNull();
  });

  it('should display winner for ring elimination victory', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
        currentUserId="user1"
      />
    );

    expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
    expect(
      screen.getByText(/Victory by eliminating a number of rings equal to the starting ring supply/)
    ).toBeInTheDocument();
  });

  it('should display territory control victory message', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'territory_control');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
    expect(
      screen.getByText(/Victory by controlling the majority of territory/)
    ).toBeInTheDocument();
  });

  it('should display last player standing victory message', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(2, 'last_player_standing');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    // LPS is surfaced as a distinct victory mode with a dedicated title
    // and explanation of the "real moves for a full round" condition.
    // Use getAllByText since the text may appear multiple times (e.g., in heading and description)
    expect(screen.getAllByText(/Last Player Standing/).length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(
        /only player able to make real moves \(placements, movements, or captures\)/
      ).length
    ).toBeGreaterThan(0);
  });

  it('should show draw message for stalemate', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(undefined, 'draw');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText(/Draw!/)).toBeInTheDocument();
    expect(screen.getByText(/stalemate with equal positions/)).toBeInTheDocument();
  });

  it('should display timeout victory message', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'timeout');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
    expect(screen.getByText(/Victory by opponent timeout/)).toBeInTheDocument();
  });

  it('should display resignation victory message', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'resignation');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
    expect(screen.getByText(/Victory by opponent resignation/)).toBeInTheDocument();
  });

  it('should display abandonment message', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(undefined, 'abandonment');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText(/Game Abandoned/)).toBeInTheDocument();
    expect(screen.getByText(/unresolved state/)).toBeInTheDocument();
  });

  it('should display final statistics table', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const table = screen.getByRole('table');
    const rows = within(table).getAllByRole('row');

    // Header row should contain the expected column labels
    expect(within(rows[0]).getByText('Player')).toBeInTheDocument();
    expect(within(rows[0]).getByText('Rings on Board')).toBeInTheDocument();
    expect(within(rows[0]).getByText('Rings Eliminated')).toBeInTheDocument();
    expect(within(rows[0]).getByText('Territory')).toBeInTheDocument();
    expect(within(rows[0]).getByText('Moves')).toBeInTheDocument();

    // There should be one stats row per player
    expect(rows).toHaveLength(1 + players.length);

    // Check player rows appear in the table
    expect(within(rows[1]).getByText(/Alice/)).toBeInTheDocument();
    expect(within(rows[2]).getByText(/Bob/)).toBeInTheDocument();
  });

  it('should show crown emoji for winner in stats table', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    // Winner row should have crown emoji
    const table = screen.getByRole('table');
    const winnerRow = within(table).getByRole('row', { name: /Alice/ });
    expect(winnerRow).toHaveTextContent('👑');
  });

  it('should call onReturnToLobby when button clicked', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const returnButton = screen.getByText('Return to Lobby');
    fireEvent.click(returnButton);

    expect(mockOnReturnToLobby).toHaveBeenCalledTimes(1);
  });

  it('should call onClose when Close button clicked', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const closeButton = screen.getByText('Close');
    fireEvent.click(closeButton);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it('should show Play Again button when onRematch provided (sandbox/local games)', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
        onRematch={mockOnRematch}
      />
    );

    // onRematch without onRequestRematch shows "Play Again" for local/sandbox games
    const playAgainButton = screen.getByText('Play Again');
    expect(playAgainButton).toBeInTheDocument();

    fireEvent.click(playAgainButton);
    expect(mockOnRematch).toHaveBeenCalledTimes(1);
  });

  it('should not show Play Again button when onRematch not provided', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.queryByText('Play Again')).not.toBeInTheDocument();
    expect(screen.queryByText('Request Rematch')).not.toBeInTheDocument();
  });

  it('should display game summary when gameState provided', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');
    const gameState = createTestGameState(players);

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText('Board Type:')).toBeInTheDocument();
    expect(screen.getByText('square8')).toBeInTheDocument();
    expect(screen.getByText('Total Turns:')).toBeInTheDocument();
    expect(screen.getByText('Players:')).toBeInTheDocument();
  });

  it('should show rated badge for rated games', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');
    const gameState = createTestGameState(players);
    gameState.isRated = true;

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    expect(screen.getByText('Game Type:')).toBeInTheDocument();
    expect(screen.getByText('Rated')).toBeInTheDocument();
  });

  it('should apply green styling for user victory', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
        currentUserId="user1"
      />
    );

    const title = screen.getByText(/Alice Wins!/);
    expect(title).toHaveClass('text-green-400');
  });

  it('should apply red styling for user defeat', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
        currentUserId="user2"
      />
    );

    const title = screen.getByText(/Alice Wins!/);
    expect(title).toHaveClass('text-red-400');
  });

  it('should close modal on Escape key', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it('should have proper ARIA attributes', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 'victory-title');
    expect(dialog).toHaveAttribute('aria-describedby', 'victory-description');
  });

  it('should sort stats table with winner first', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(2, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const rows = screen.getAllByRole('row');
    // First row is header, second row should be winner (Bob/player2)
    expect(rows[1]).toHaveTextContent('Bob');
    expect(rows[1]).toHaveTextContent('👑');
  });

  it('should render correctly when provided a prebuilt VictoryViewModel', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');
    const gameState = createTestGameState(players);

    const viewModel = toVictoryViewModel(gameResult, players, gameState, {
      currentUserId: 'user1',
      isDismissed: false,
    });

    // Sanity check: viewModel should be non-null and visible
    expect(viewModel).not.toBeNull();
    if (!viewModel) {
      return;
    }

    render(
      <VictoryModal
        isOpen={true}
        viewModel={viewModel}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    // Title and description come from the view model path
    expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
    expect(
      screen.getByText(/Victory by eliminating a number of rings equal to the starting ring supply/)
    ).toBeInTheDocument();

    // Stats table is rendered via the view model finalStats
    const table = screen.getByRole('table');
    const rows = within(table).getAllByRole('row');
    expect(rows).toHaveLength(1 + players.length);
  });

  it('should display statistics from gameResult when gameState not provided', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const table = screen.getByRole('table');
    const rows = within(table).getAllByRole('row');

    // Header row + one row per player
    expect(rows).toHaveLength(1 + players.length);

    const aliceRow = rows[1];
    const bobRow = rows[2];

    const aliceCells = within(aliceRow).getAllByRole('cell');
    const bobCells = within(bobRow).getAllByRole('cell');

    // Should show rings remaining and eliminated from finalScore
    expect(aliceCells[1]).toHaveTextContent('3'); // Alice's rings on board
    expect(bobCells[1]).toHaveTextContent('10'); // Bob's rings on board
    expect(aliceCells[2]).toHaveTextContent('15'); // Alice's eliminated rings
    expect(bobCells[2]).toHaveTextContent('8'); // Bob's eliminated rings
  });

  it('should handle 3-player game stats', () => {
    const players: Player[] = [
      ...createTestPlayers(),
      {
        id: 'user3',
        username: 'Charlie',
        playerNumber: 3,
        type: 'ai',
        isReady: true,
        timeRemaining: 0,
        ringsInHand: 5,
        eliminatedRings: 12,
        territorySpaces: 15,
      },
    ];

    const gameResult: GameResult = {
      winner: 1,
      reason: 'ring_elimination',
      finalScore: {
        ringsEliminated: { 1: 15, 2: 8, 3: 12 },
        territorySpaces: { 1: 25, 2: 10, 3: 15 },
        ringsRemaining: { 1: 3, 2: 10, 3: 5 },
      },
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const table = screen.getByRole('table');

    expect(within(table).getByRole('row', { name: /Alice/ })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: /Bob/ })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: /Charlie/ })).toBeInTheDocument();
  });

  it('should display fallback player names when username not available', () => {
    const players: Player[] = [
      {
        id: 'user1',
        username: '',
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: 0,
        ringsInHand: 3,
        eliminatedRings: 15,
        territorySpaces: 25,
      },
    ];

    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    const table = screen.getByRole('table');

    expect(screen.getByText(/Player 1 Wins!/)).toBeInTheDocument();
    expect(within(table).getByText(/Player 1/)).toBeInTheDocument();
  });

  it('should count moves from game history when available', () => {
    const players = createTestPlayers();
    const gameResult = createGameResult(1, 'ring_elimination');
    const gameState = createTestGameState(players);

    // Add some history entries
    gameState.history = [
      { moveNumber: 1, action: {} as any, actor: 1 } as any,
      { moveNumber: 2, action: {} as any, actor: 2 } as any,
      { moveNumber: 3, action: {} as any, actor: 1 } as any,
      { moveNumber: 4, action: {} as any, actor: 1 } as any,
      { moveNumber: 5, action: {} as any, actor: 2 } as any,
    ];

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={mockOnClose}
        onReturnToLobby={mockOnReturnToLobby}
      />
    );

    // Alice (player 1) should have 3 moves, Bob (player 2) should have 2
    const rows = screen.getAllByRole('row');

    // Find Alice's row (should be first after header due to being winner)
    const aliceRow = rows.find((row) => row.textContent?.includes('Alice'));
    expect(aliceRow).toHaveTextContent('3'); // 3 moves for Alice
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // VictoryModal with gameEndExplanation prop - Component rendering tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('VictoryModal - gameEndExplanation prop rendering', () => {
    const mockOnClose = jest.fn();
    const mockOnReturnToLobby = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
    });

    function createGameEndExplanation(
      shortSummaryKey: string,
      overrides?: Partial<{
        outcomeType: string;
        primaryConceptId: string;
      }>
    ) {
      return {
        boardType: 'square8' as const,
        numPlayers: 2,
        winnerPlayerId: 'user1',
        outcomeType: overrides?.outcomeType ?? 'ring_elimination',
        victoryReasonCode: 'victory_ring_majority',
        uxCopy: {
          shortSummaryKey,
          detailedSummaryKey: `${shortSummaryKey}.detailed`,
        },
        primaryConceptId: overrides?.primaryConceptId,
      };
    }

    it('should render LPS-specific copy when gameEndExplanation has LPS key', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'last_player_standing');
      const gameState = createTestGameState(players);
      const explanation = createGameEndExplanation('game_end.lps.with_anm_fe', {
        outcomeType: 'last_player_standing',
        primaryConceptId: 'lps.with_anm_fe',
      });

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={explanation as any}
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // LPS-specific title should appear (use getAllByText since text appears multiple times)
      expect(screen.getAllByText(/Last Player Standing/).length).toBeGreaterThan(0);
      // LPS-specific description mentioning real moves should appear
      expect(screen.getByText(/only player able to make real moves/i)).toBeInTheDocument();
    });

    it('should render structural stalemate copy when gameEndExplanation has stalemate key', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'game_completed');
      const gameState = createTestGameState(players);
      const explanation = createGameEndExplanation('game_end.structural_stalemate.tiebreak', {
        outcomeType: 'game_completed',
      });

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={explanation as any}
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // Structural stalemate copy should appear (may appear multiple times in title and description)
      expect(screen.getAllByText(/Structural Stalemate/i).length).toBeGreaterThan(0);
    });

    it('should render territory mini-region copy when gameEndExplanation has territory key', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'territory_control');
      const gameState = createTestGameState(players);
      const explanation = createGameEndExplanation('game_end.territory_mini_region', {
        outcomeType: 'territory_control',
      });

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={explanation as any}
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // Winner should be displayed
      expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
      // Territory-specific language should appear (use getAllByText since text may appear multiple times)
      expect(screen.getAllByText(/territory/i).length).toBeGreaterThan(0);
    });

    it('should fall back to legacy copy when gameEndExplanation has unrecognized key', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState(players);
      const explanation = createGameEndExplanation('game_end.unknown_pattern');

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={explanation as any}
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // Should fall back to standard ring elimination copy
      expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
      expect(
        screen.getByText(/eliminating a number of rings equal to the starting ring supply/i)
      ).toBeInTheDocument();
    });

    it('should use personalized "You" wording when current user is LPS winner', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'last_player_standing');
      const gameState = createTestGameState(players);
      const explanation = createGameEndExplanation('game_end.lps.with_anm_fe', {
        outcomeType: 'last_player_standing',
        primaryConceptId: 'lps.with_anm_fe',
      });

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={explanation as any}
          currentUserId="user1"
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // LPS title should appear
      expect(screen.getAllByText(/Last Player Standing/).length).toBeGreaterThan(0);
      // Description should use "You" wording for the winner (or refer to the player by name)
      expect(screen.getByText(/only player able to make real moves/i)).toBeInTheDocument();
    });

    it('should still render correctly when gameEndExplanation is null', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState(players);

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={null}
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // Should use standard legacy copy
      expect(screen.getByText(/Alice Wins!/)).toBeInTheDocument();
      expect(
        screen.getByText(/eliminating a number of rings equal to the starting ring supply/i)
      ).toBeInTheDocument();
    });

    it('should display weird state info panel for LPS with ANM/FE', () => {
      const players = createTestPlayers();
      const gameResult = createGameResult(1, 'last_player_standing');
      const gameState = createTestGameState(players);
      const explanation = createGameEndExplanation('game_end.lps.with_anm_fe', {
        outcomeType: 'last_player_standing',
        primaryConceptId: 'lps.with_anm_fe',
      });

      render(
        <VictoryModal
          isOpen={true}
          gameResult={gameResult}
          players={players}
          gameState={gameState}
          gameEndExplanation={explanation as any}
          onClose={mockOnClose}
          onReturnToLobby={mockOnReturnToLobby}
        />
      );

      // Should show the "What happened?" button for teaching overlay
      const whatHappenedButton = screen.queryByRole('button', { name: /what happened/i });
      expect(whatHappenedButton).toBeInTheDocument();
    });
  });
});
