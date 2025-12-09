import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { VictoryModal } from '../../../src/client/components/VictoryModal';
import type { GameResult, Player, GameState } from '../../../src/shared/types/game';
import type { GameEndExplanation } from '../../../src/shared/engine/gameEndExplanation';
import type { VictoryViewModel } from '../../../src/client/adapters/gameViewModels';

function createPlayers(): Player[] {
  return [
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
    victoryThreshold: 19,
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
    // The full detailed copy includes additional context about what happened
    expect(
      screen.getByText(
        /Victory by Territory Control after resolving the final disconnected mini-region.*Processing that region/i
      )
    ).toBeInTheDocument();
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
      screen.getByText(/Victory by eliminating more than half of all rings/i)
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

  it('shows "What happened?" link for territory mini-region explanation', async () => {
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

    const helpLink = await screen.findByRole('button', { name: /What happened\?/i });
    expect(helpLink).toBeInTheDocument();

    await userEvent.click(helpLink);

    await waitFor(() => {
      // TeachingOverlay for the territory topic should be open with a
      // heading matching the canonical "Territory" teaching topic.
      const territoryHeading = screen.getByRole('heading', { name: /Territory$/i });
      expect(territoryHeading).toBeInTheDocument();
    });
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
