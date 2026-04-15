import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { VictoryModal } from '../../../src/client/components/VictoryModal';
import type { GameResult, Player, GameState } from '../../../src/shared/types/game';
import type { GameEndExplanation } from '../../../src/shared/engine/gameEndExplanation';
import type { VictoryViewModel } from '../../../src/client/adapters/gameViewModels';

// Silence jsdom XHR noise from confetti/assets in VictoryModal during tests.
let consoleErrorSpy: jest.SpyInstance;
beforeAll(() => {
  consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

function createPlayers(overrides: Partial<Player>[] = []): Player[] {
  const players: Player[] = [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 5,
      eliminatedRings: 10,
      territorySpaces: 8,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 3,
      eliminatedRings: 8,
      territorySpaces: 4,
    },
  ];

  return players.map((player, index) => ({ ...player, ...(overrides[index] ?? {}) }));
}

function createGameState(players: Player[]): GameState {
  return {
    id: 'game-1',
    boardType: 'square8',
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: { 1: 10, 2: 8 },
      size: 8,
      type: 'square8',
    },
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'completed',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 36,
    totalRingsEliminated: 18,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
  };
}

function createGameResult(winner: number | undefined, reason: GameResult['reason']): GameResult {
  return {
    winner,
    reason,
    finalScore: {
      ringsEliminated: { 1: 10, 2: 8 },
      territorySpaces: { 1: 8, 2: 4 },
      ringsRemaining: { 1: 5, 2: 3 },
    },
  };
}

describe('VictoryModal – basic rendering', () => {
  it('does not render when isOpen is false', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={false}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('renders when isOpen is true with gameResult', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/Alice Wins/i)).toBeInTheDocument();
  });

  it('displays player stats in the final stats table', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();
    // Check table headers
    expect(screen.getByText('Rings on Board')).toBeInTheDocument();
    expect(screen.getByText('Rings Eliminated')).toBeInTheDocument();
    expect(screen.getByText('Territory')).toBeInTheDocument();
  });

  it('renders sandbox match summary stats above the final table when provided', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameDurationMs={272000}
        aiAverageThinkTimeMsByPlayer={{ 2: 1200 }}
        sessionRecord={{ wins: 2, losses: 1 }}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('Game duration:')).toBeInTheDocument();
    expect(screen.getByText('4m 32s')).toBeInTheDocument();
    expect(screen.getByText('AI avg think time:')).toBeInTheDocument();
    expect(screen.getByText('1.2s')).toBeInTheDocument();
    expect(screen.getByText('Session:')).toBeInTheDocument();
    expect(screen.getByText('2W-1L')).toBeInTheDocument();
  });

  it('shows per-player AI average think times when multiple AI seats are timed', () => {
    const players = createPlayers([{ username: 'Oracle' }, { username: 'Sentinel' }]);
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        aiAverageThinkTimeMsByPlayer={{ 1: 1200, 2: 800 }}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('AI avg think time:')).toBeInTheDocument();
    expect(screen.getAllByText('Oracle').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Sentinel').length).toBeGreaterThan(0);
    expect(screen.getByText('1.2s')).toBeInTheDocument();
    expect(screen.getByText('800ms')).toBeInTheDocument();
  });
});

describe('VictoryModal – GameEndExplanation-driven copy', () => {
  it('uses LPS-specific copy when explanation has LPS with ANM/FE key', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'last_player_standing');
    const explanation: GameEndExplanation = {
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      primaryConceptId: 'lps_real_actions',
      uxCopy: {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'],
        rulesContextTags: ['anm_forced_elimination'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        currentUserId="p2"
      />
    );

    expect(screen.getByText('👑 Last Player Standing')).toBeInTheDocument();
    expect(
      screen.getByText(/Alice was the only player able to make real moves/i)
    ).toBeInTheDocument();
  });

  it('uses structural stalemate copy when explanation has stalemate key', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'game_completed');
    const explanation: GameEndExplanation = {
      outcomeType: 'structural_stalemate',
      victoryReasonCode: 'victory_structural_stalemate_tiebreak',
      primaryConceptId: 'structural_stalemate',
      uxCopy: {
        shortSummaryKey: 'game_end.structural_stalemate.short',
        detailedSummaryKey: 'game_end.structural_stalemate.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['STRUCTURAL_STALEMATE_TIEBREAK'],
        rulesContextTags: ['structural_stalemate'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('🧱 Structural Stalemate')).toBeInTheDocument();
    expect(screen.getByText(/The game reached a structural stalemate/i)).toBeInTheDocument();
  });

  it('uses territory mini-region copy when explanation has mini-region key', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'territory_control');
    const explanation: GameEndExplanation = {
      outcomeType: 'territory_control',
      victoryReasonCode: 'victory_territory_majority',
      primaryConceptId: 'territory_mini_regions',
      uxCopy: {
        shortSummaryKey: 'game_end.territory_mini_region.short',
        detailedSummaryKey: 'game_end.territory_mini_region.detailed',
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('🏰 Alice Wins!')).toBeInTheDocument();

    // Assert mini-region semantics without pinning exact prose.
    const description = screen.getByText(/final disconnected mini-region/i);
    expect(description).toBeInTheDocument();
    expect(description).toHaveTextContent(/Territory Control/i);
    expect(description).toHaveTextContent(/rules compared Territory spaces/i);
  });

  it('falls back to legacy copy for unrecognized uxCopy key', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');
    const explanation: GameEndExplanation = {
      outcomeType: 'ring_elimination',
      victoryReasonCode: 'victory_ring_majority',
      primaryConceptId: 'ring_majority',
      uxCopy: {
        shortSummaryKey: 'unknown.key',
        detailedSummaryKey: 'unknown.key',
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    // Falls back to ring elimination copy based on gameResult.reason
    expect(screen.getByText(/Alice Wins/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        /Victory by Ring Elimination: eliminated rings reached the victory threshold/i
      )
    ).toBeInTheDocument();
  });

  it('falls back to legacy copy when explanation is null', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'last_player_standing');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={null}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('👑 Last Player Standing')).toBeInTheDocument();
  });

  it('uses "You" wording when current user is the winner with LPS explanation', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'last_player_standing');
    const explanation: GameEndExplanation = {
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      primaryConceptId: 'lps_real_actions',
      uxCopy: {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        currentUserId="p1" // Winner perspective
      />
    );

    expect(
      screen.getByText(/You were the only player able to make real moves/i)
    ).toBeInTheDocument();
  });
});

describe('VictoryModal – view model props', () => {
  it('renders from pre-transformed viewModel prop', () => {
    const viewModel: VictoryViewModel = {
      isVisible: true,
      title: '🏆 Custom Title',
      description: 'Custom description from view model',
      titleColorClass: 'text-green-400',
      winner: {
        id: 'p1',
        playerNumber: 1,
        username: 'Alice',
        isCurrentPlayer: false,
        isUserPlayer: true,
        colorClass: 'bg-blue-500',
        ringStats: { inHand: 5, onBoard: 10, eliminated: 3, total: 18 },
        territorySpaces: 8,
        aiInfo: { isAI: false },
      },
      finalStats: [
        {
          player: {
            id: 'p1',
            playerNumber: 1,
            username: 'Alice',
            isCurrentPlayer: false,
            isUserPlayer: true,
            colorClass: 'bg-blue-500',
            ringStats: { inHand: 5, onBoard: 10, eliminated: 3, total: 18 },
            territorySpaces: 8,
            aiInfo: { isAI: false },
          },
          ringsOnBoard: 10,
          ringsEliminated: 3,
          territorySpaces: 8,
          totalMoves: 25,
          isWinner: true,
        },
      ],
      gameSummary: {
        boardType: 'square8',
        totalTurns: 25,
        playerCount: 2,
        isRated: false,
      },
      userWon: true,
      userLost: false,
      isDraw: false,
    };

    render(
      <VictoryModal
        isOpen={true}
        viewModel={viewModel}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('🏆 Custom Title')).toBeInTheDocument();
    expect(screen.getByText('Custom description from view model')).toBeInTheDocument();
  });
});

describe('VictoryModal – weird state teaching link', () => {
  it('shows "What happened?" link for LPS with ANM/FE explanation', async () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'last_player_standing');
    const explanation: GameEndExplanation = {
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      primaryConceptId: 'lps_real_actions',
      uxCopy: {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'],
        rulesContextTags: ['anm_forced_elimination'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    const helpLink = await screen.findByRole('button', { name: /What happened\?/i });
    expect(helpLink).toBeInTheDocument();
  });

  it('shows "What happened?" link for structural stalemate explanation', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'game_completed');
    const explanation: GameEndExplanation = {
      outcomeType: 'structural_stalemate',
      victoryReasonCode: 'victory_structural_stalemate_tiebreak',
      primaryConceptId: 'structural_stalemate',
      uxCopy: {
        shortSummaryKey: 'game_end.structural_stalemate.short',
        detailedSummaryKey: 'game_end.structural_stalemate.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['STRUCTURAL_STALEMATE_TIEBREAK'],
        rulesContextTags: ['structural_stalemate'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    const helpLink = screen.getByRole('button', { name: /What happened\?/i });
    expect(helpLink).toBeInTheDocument();
  });

  it('suppresses "What happened?" link for territory ANM no-action explanations', async () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'territory_control');
    const explanation: GameEndExplanation = {
      outcomeType: 'territory_control',
      victoryReasonCode: 'victory_territory_majority',
      primaryConceptId: 'territory_mini_regions',
      uxCopy: {
        shortSummaryKey: 'game_end.territory_mini_region.short',
        detailedSummaryKey: 'game_end.territory_mini_region.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['ANM_TERRITORY_NO_ACTIONS'],
        primaryReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
        rulesContextTags: ['territory_mini_region'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.queryByRole('button', { name: /What happened\?/i })).toBeNull();
  });

  it('shows forced-elimination weird-state banner copy for ANM/LPS explanation', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'last_player_standing');
    const explanation: GameEndExplanation = {
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      primaryConceptId: 'lps_real_actions',
      uxCopy: {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'],
        primaryReasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        rulesContextTags: ['anm_forced_elimination'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText(/What happened\?/i)).toBeInTheDocument();
    expect(screen.getByText(/only player able to make real moves/i)).toBeInTheDocument();
  });
});

describe('VictoryModal – action buttons', () => {
  it('calls onClose when Close button is clicked', async () => {
    const onClose = jest.fn();
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={onClose}
        onReturnToLobby={jest.fn()}
      />
    );

    const closeButton = screen.getByRole('button', { name: /Close/i });
    await userEvent.click(closeButton);

    expect(onClose).toHaveBeenCalled();
  });

  it('calls onReturnToLobby when Return to Lobby button is clicked', async () => {
    const onReturnToLobby = jest.fn();
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={onReturnToLobby}
      />
    );

    const lobbyButton = screen.getByRole('button', { name: /Return to Lobby/i });
    await userEvent.click(lobbyButton);

    expect(onReturnToLobby).toHaveBeenCalled();
  });

  it('shows Play Again button when onRematch is provided', () => {
    const onRematch = jest.fn();
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRematch={onRematch}
      />
    );

    const rematchButton = screen.getByRole('button', { name: /Play Again/i });
    expect(rematchButton).toBeInTheDocument();
  });

  it('shows a training availability note when sharing is unavailable', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onSubmitForTraining={jest.fn()}
        trainingSubmission={{
          isAvailable: false,
          isSubmitting: false,
          wasSubmitted: false,
          error: null,
          availabilityNote:
            'This environment does not have the replay service configured, so this win cannot be queued for training review.',
        }}
      />
    );

    expect(screen.getByText('Training review unavailable')).toBeInTheDocument();
    expect(
      screen.getByText(
        /This environment does not have the replay service configured, so this win cannot be queued for training review/i
      )
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /Share This Win for Training/i })
    ).not.toBeInTheDocument();
  });
});

describe('VictoryModal – draw result', () => {
  it('displays draw message when result is a draw', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(undefined, 'draw');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(screen.getByText('🤝 Draw!')).toBeInTheDocument();
    expect(screen.getByText(/ended in a stalemate/i)).toBeInTheDocument();
  });
});

describe('VictoryModal – rematch states', () => {
  it('shows accepted message when rematch is accepted', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRequestRematch={jest.fn()}
        rematchStatus={{ status: 'accepted', isPending: false, isRequester: false }}
      />
    );

    expect(screen.getByText('Rematch on! Joining new game...')).toBeInTheDocument();
  });

  it('shows declined message when rematch is declined', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRequestRematch={jest.fn()}
        rematchStatus={{ status: 'declined', isPending: false, isRequester: false }}
      />
    );

    expect(screen.getByText('Opponent declined the rematch')).toBeInTheDocument();
  });

  it('shows expired message with request button when rematch expired', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');
    const onRequestRematch = jest.fn();

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRequestRematch={onRequestRematch}
        rematchStatus={{ status: 'expired', isPending: false, isRequester: false }}
      />
    );

    expect(screen.getByText('Rematch offer expired')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Play Again\?/i })).toBeInTheDocument();
  });

  it('shows accept/decline buttons when opponent requests rematch', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');
    const onAcceptRematch = jest.fn();
    const onDeclineRematch = jest.fn();
    // Set expires 30 seconds in the future
    const expiresAt = new Date(Date.now() + 30000).toISOString();

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRequestRematch={jest.fn()}
        onAcceptRematch={onAcceptRematch}
        onDeclineRematch={onDeclineRematch}
        rematchStatus={{
          status: 'pending',
          isPending: true,
          isRequester: false,
          requestId: 'req-123',
          requesterUsername: 'OpponentPlayer',
          expiresAt,
        }}
      />
    );

    expect(screen.getByText('OpponentPlayer wants a rematch!')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Accept/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Decline/i })).toBeInTheDocument();
  });

  it('shows waiting message when current user requested rematch', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');
    const expiresAt = new Date(Date.now() + 30000).toISOString();

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRequestRematch={jest.fn()}
        rematchStatus={{
          status: 'pending',
          isPending: true,
          isRequester: true,
          expiresAt,
        }}
      />
    );

    expect(screen.getByText('Waiting for opponent...')).toBeInTheDocument();
  });

  it('shows request rematch button when no pending request', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'ring_elimination');
    const onRequestRematch = jest.fn();

    render(
      <VictoryModal
        isOpen={true}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
        onRequestRematch={onRequestRematch}
      />
    );

    expect(screen.getByRole('button', { name: /Play Again\?/i })).toBeInTheDocument();
  });

  it('invokes telemetry-friendly teaching overlay from weird-state link and closes modal', async () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult(1, 'last_player_standing');
    const onClose = jest.fn();
    const explanation: GameEndExplanation = {
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      primaryConceptId: 'lps_real_actions',
      uxCopy: {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
      weirdStateContext: {
        reasonCodes: ['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'],
        primaryReasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        rulesContextTags: ['anm_forced_elimination'],
      },
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'p1',
    };

    render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={onClose}
        onReturnToLobby={jest.fn()}
      />
    );

    const helpLink = await screen.findByRole('button', { name: /What happened\?/i });
    expect(helpLink).toBeInTheDocument();

    await userEvent.click(helpLink);

    // Closing the modal should still work after invoking the teaching overlay entrypoint.
    const closeButtons = screen.getAllByRole('button', { name: /Close/i });
    await userEvent.click(closeButtons[0]);
    expect(onClose).toHaveBeenCalled();
  });
});
