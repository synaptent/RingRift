import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../src/client/components/GameHUD';
import { toHUDViewModel } from '../../src/client/adapters/gameViewModels';
import { GameState, Player, BoardState, PlayerChoice } from '../../src/shared/types/game';
import type { GameEndExplanation } from '../../src/shared/engine/gameEndExplanation';

// Helper to create a minimal test GameState
function createTestGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultBoard: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  const defaultPlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
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
      username: 'Player 2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  return {
    id: 'test-game',
    boardType: 'square8',
    board: defaultBoard,
    players: defaultPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
    ...overrides,
  };
}

describe('GameHUD', () => {
  it('should display phase indicator', () => {
    const gameState = createTestGameState({ currentPhase: 'movement' });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    expect(screen.getByText('Movement Phase')).toBeInTheDocument();
    expect(
      screen.getByText('Select one of your stacks and move it to a legal destination.')
    ).toBeInTheDocument();
  });

  it('should highlight current player', () => {
    const gameState = createTestGameState({ currentPlayer: 1 });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} currentUserId="p1" />);

    expect(screen.getByText('Current Turn')).toBeInTheDocument();
  });

  it('should display ring statistics', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    // There is a ring stats block for each player, so these labels may
    // appear multiple times. We only assert that at least one is present.
    expect(screen.getAllByText('In Hand').length).toBeGreaterThan(0);
    expect(screen.getAllByText('On Board').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Rings Eliminated').length).toBeGreaterThan(0);
  });

  it('should show timer when time controls active', () => {
    const gameState = createTestGameState({
      timeControl: { type: 'rapid', initialTime: 300000, increment: 5000 },
    });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    // Both players have timers; assert that at least one mm:ss time is shown.
    expect(screen.getAllByText(/\d+:\d{2}/).length).toBeGreaterThan(0);
  });

  it('should display game progress turn counter', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    expect(screen.getByText('Turn')).toBeInTheDocument();
  });

  it('should show AI badge for AI players', () => {
    const gameState = createTestGameState({
      players: [
        {
          id: 'p1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'ai-1',
          username: 'AI Level 5',
          playerNumber: 2,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 5,
          aiProfile: { difficulty: 5, aiType: 'heuristic' },
        },
      ],
    });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    expect(screen.getByText('🤖 AI')).toBeInTheDocument();
  });

  it('should display connection status', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(
      <GameHUD gameState={gameState} currentPlayer={currentPlayer} connectionStatus="connected" />
    );

    expect(screen.getByText(/Connected/)).toBeInTheDocument();
  });

  it('should render reconnecting and disconnected connection statuses (legacy props path)', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    const { rerender } = render(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        connectionStatus="reconnecting"
      />
    );

    expect(screen.getByText(/Reconnecting…/)).toBeInTheDocument();

    rerender(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        connectionStatus="disconnected"
      />
    );

    expect(screen.getByText(/Disconnected/)).toBeInTheDocument();
  });

  it('should show spectator badge when spectating', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} isSpectator={true} />);

    expect(screen.getByText('Spectator Mode')).toBeInTheDocument();
  });

  it('should display instruction when provided', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        instruction="Place a ring on an empty edge space."
      />
    );

    expect(screen.getByText('Place a ring on an empty edge space.')).toBeInTheDocument();
  });

  it('should show territory count when player has territory', () => {
    const gameState = createTestGameState({
      players: [
        {
          id: 'p1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 5,
        },
        {
          id: 'p2',
          username: 'Player 2',
          playerNumber: 2,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
    });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    // Territory summary text should be rendered for the player with territory.
    expect(screen.getByText(/territory space/)).toBeInTheDocument();
  });

  it('should display different phase colors and icons', () => {
    const phases: Array<GameState['currentPhase']> = [
      'ring_placement',
      'movement',
      'chain_capture',
      'line_processing',
      'territory_processing',
    ];

    phases.forEach((phase) => {
      const gameState = createTestGameState({ currentPhase: phase });
      const currentPlayer = gameState.players[0];

      const { container } = render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

      // Should have phase-specific styling
      const phaseIndicator = container.querySelector('[class*="bg-"]');
      expect(phaseIndicator).toBeInTheDocument();
    });
  });

  it('should surface all three victory modes copy', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    const helper = screen.getByTestId('victory-conditions-help');
    expect(helper).toBeInTheDocument();
    expect(screen.getByText(/Ring Elimination – Win by eliminating/)).toBeInTheDocument();
    expect(screen.getByText(/Territory Control – Win by controlling/)).toBeInTheDocument();
    expect(screen.getByText(/Last Player Standing/)).toBeInTheDocument();
  });

  it('renders contextual tooltips for each victory condition', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    const eliminationTrigger = screen.getByTestId('victory-tooltip-elimination-trigger');
    const territoryTrigger = screen.getByTestId('victory-tooltip-territory-trigger');
    const lastPlayerTrigger = screen.getByTestId('victory-tooltip-last-player-standing-trigger');

    // Elimination tooltip
    fireEvent.mouseEnter(eliminationTrigger);
    let tooltip = screen.getByRole('tooltip');
    expect(tooltip).toHaveTextContent(
      /You win Ring Elimination when the rings you have eliminated exceed 50% of all rings in the game/
    );
    fireEvent.mouseLeave(eliminationTrigger);
    expect(screen.queryByRole('tooltip')).toBeNull();

    // Territory tooltip
    fireEvent.mouseEnter(territoryTrigger);
    tooltip = screen.getByRole('tooltip');
    expect(tooltip).toHaveTextContent('Territory spaces are collapsed cells you permanently own.');
    fireEvent.mouseLeave(territoryTrigger);
    expect(screen.queryByRole('tooltip')).toBeNull();

    // Last Player Standing tooltip
    fireEvent.mouseEnter(lastPlayerTrigger);
    tooltip = screen.getByRole('tooltip');
    expect(tooltip).toHaveTextContent(
      'Last Player Standing requires three consecutive full rounds'
    );
    fireEvent.mouseLeave(lastPlayerTrigger);
    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  describe('spectator banner (view model path)', () => {
    it('renders spectator mode banner with accessible status when spectating', () => {
      const gameState = createTestGameState({
        spectators: ['spectator-1'],
      });

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: true,
        currentUserId: 'spectator-1',
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      // Query the specific spectator status element by its accessible name (multiple elements now have role="status" after ARIA updates)
      const banner = screen.getByRole('status', { name: /Spectator Mode/i });
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveAttribute('aria-live', 'polite');
      expect(screen.getByText('You are watching this game')).toBeInTheDocument();
      expect(screen.getByText('1 viewer total')).toBeInTheDocument();
    });

    it('renders phase tooltip with spectator-focused copy when spectating', () => {
      const gameState = createTestGameState({
        spectators: ['spectator-1'],
        currentPhase: 'ring_placement',
      });

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: true,
        currentUserId: 'spectator-1',
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const trigger = screen.getByTestId('phase-tooltip-trigger');
      fireEvent.mouseEnter(trigger);

      const tooltip = screen.getByRole('tooltip');
      expect(tooltip).toHaveTextContent('Place your rings on the board to build stacks');
      expect(tooltip).toHaveTextContent('Spectators: Player is placing rings on the board');
    });

    it('renders spectator count chip with accessible label for non-spectator viewers', () => {
      const gameState = createTestGameState({
        spectators: ['spectator-1', 'spectator-2'],
      });

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const chip = screen.getByRole('status', { name: /2 spectators/ });
      expect(chip).toBeInTheDocument();
      expect(chip).toHaveAttribute('title', '2 people watching');
      expect(screen.getByText('2 watching')).toBeInTheDocument();
    });
  });

  describe('decision phase banner (view model path)', () => {
    it('renders acting-player decision banner with countdown and warning styling', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-1',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
        timeoutMs: 30000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now() + 4000,
        choiceTimeRemainingMs: 4000,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const banner = screen.getByTestId('decision-phase-banner');
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveTextContent('Your decision:');
      expect(banner).toHaveTextContent('Choose Line Reward');

      // Countdown label should be visible with low-time warning styling.
      const timerLabel = screen.getByText('0:04');
      expect(timerLabel).toBeInTheDocument();

      const countdownPill = screen.getByTestId('decision-phase-countdown');
      expect(countdownPill).toHaveAttribute('data-severity', 'warning');
      expect(countdownPill).not.toHaveAttribute('data-server-capped');
    });

    it('applies server-capped styling and attributes when the decision countdown is capped by the server', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-server-capped',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
        timeoutMs: 30000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now() + 15_000,
        // Server has capped the effective countdown below the baseline 15s
        choiceTimeRemainingMs: 3_000,
        decisionIsServerCapped: true,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const countdownPill = screen.getByTestId('decision-phase-countdown');
      expect(countdownPill).toHaveAttribute('data-severity', 'critical');
      expect(countdownPill).toHaveAttribute('data-server-capped', 'true');
      expect(countdownPill).toHaveTextContent('Server deadline');
    });

    it('applies normal severity when more than 10 seconds remain', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-normal',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
        timeoutMs: 30_000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now() + 11_000,
        choiceTimeRemainingMs: 11_000,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const countdownPill = screen.getByTestId('decision-phase-countdown');
      expect(countdownPill).toHaveAttribute('data-severity', 'normal');
      expect(countdownPill).not.toHaveAttribute('data-server-capped');
    });

    it('applies warning severity exactly at the 10 second threshold', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-warning-threshold',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
        timeoutMs: 30_000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now() + 10_000,
        choiceTimeRemainingMs: 10_000,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const countdownPill = screen.getByTestId('decision-phase-countdown');
      expect(countdownPill).toHaveAttribute('data-severity', 'warning');
      expect(countdownPill).not.toHaveAttribute('data-server-capped');
    });

    it('applies critical severity when time has fully elapsed', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-zero',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
        timeoutMs: 30_000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now(),
        choiceTimeRemainingMs: 0,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const countdownPill = screen.getByTestId('decision-phase-countdown');
      expect(countdownPill).toHaveAttribute('data-severity', 'critical');
    });

    it('renders a HUD-level decision time-pressure chip with severity and copy', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-hud-pressure',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
        timeoutMs: 30_000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now() + 4_000,
        choiceTimeRemainingMs: 4_000,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const chip = screen.getByTestId('hud-decision-time-pressure');
      expect(chip).toBeInTheDocument();
      expect(chip).toHaveAttribute('data-severity', 'warning');
      expect(chip).toHaveTextContent('Your decision timer: 0:04');
    });

    it('renders spectator-oriented decision banner when user is not acting player', () => {
      const gameState = createTestGameState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'choice-2',
        type: 'line_reward_option',
        playerNumber: 1,
        options: ['add_ring', 'add_stack'] as any,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'reconnecting',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p2',
        pendingChoice: choice,
        choiceDeadline: null,
        choiceTimeRemainingMs: null,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      // Connection banner should reflect reconnecting status while the
      // decision banner continues to surface spectator-oriented copy.
      expect(screen.getByText(/Connection: Reconnecting…/)).toBeInTheDocument();

      const banner = screen.getByTestId('decision-phase-banner');
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveTextContent('Waiting for');
      expect(banner).toHaveTextContent('to choose a line reward option');
      // No countdown label when timeRemainingMs is null.
      expect(screen.queryByLabelText('Decision timer')).toBeNull();
    });

    it('does not render decision banner when decisionPhase is absent', () => {
      const gameState = createTestGameState({
        currentPhase: 'movement',
      });

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      expect(screen.queryByTestId('decision-phase-banner')).toBeNull();
    });

    it('renders territory region-order decision banner for territory_processing', () => {
      const gameState = createTestGameState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const choice: PlayerChoice = {
        id: 'territory-choice-1',
        type: 'region_order',
        playerNumber: 1,
        options: [{}] as any,
        timeoutMs: 30_000,
      } as any;

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        pendingChoice: choice,
        choiceDeadline: Date.now() + 15_000,
        choiceTimeRemainingMs: 15_000,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const banner = screen.getByTestId('decision-phase-banner');
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveTextContent('Your decision:');
      expect(banner).toHaveTextContent('Choose Territory Region');
    });

    it('renders phase tooltip with action-focused copy for the active player', () => {
      const gameState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
      });

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      const trigger = screen.getByTestId('phase-tooltip-trigger');
      fireEvent.mouseEnter(trigger);

      const tooltip = screen.getByRole('tooltip');
      expect(tooltip).toHaveTextContent('Move a stack or initiate a capture');
      expect(tooltip).toHaveTextContent(
        'On your turn: Select your stack, then click a destination'
      );
    });

    it('renders structural-stalemate weird-state banner using explanation-driven copy', () => {
      const gameState = createTestGameState({
        gameStatus: 'finished',
      });

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

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
        isSpectator: false,
        currentUserId: 'p1',
        gameEndExplanation: explanation,
      });

      render(<GameHUD viewModel={hudViewModel} timeControl={gameState.timeControl} />);

      expect(screen.getByText(/Structural stalemate/i)).toBeInTheDocument();
      expect(
        screen.getByText(
          /No legal placements, movements, captures, or forced eliminations remain for any player/i
        )
      ).toBeInTheDocument();
      expect(screen.getByTestId('hud-weird-state-help')).toBeInTheDocument();
    });

    it('renders sandbox local-only banner when flagged by host', () => {
      const gameState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
      });

      const hudViewModel = toHUDViewModel(gameState, {
        instruction: undefined,
        connectionStatus: 'disconnected',
        lastHeartbeatAt: null,
        isSpectator: false,
        currentUserId: 'p1',
      });

      render(
        <GameHUD
          viewModel={hudViewModel}
          timeControl={gameState.timeControl}
          isLocalSandboxOnly={true}
        />
      );

      const banner = screen.getByTestId('sandbox-local-only-banner');
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveTextContent(
        "You're not logged in; this game runs as a local sandbox only."
      );
    });
  });
});
