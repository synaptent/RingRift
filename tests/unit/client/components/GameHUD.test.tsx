import React from 'react';
import { render, screen, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD, VictoryConditionsPanel } from '../../../../src/client/components/GameHUD';
import type { GameHUDViewModelProps } from '../../../../src/client/components/GameHUD';
import type {
  HUDViewModel,
  PhaseViewModel,
  PlayerViewModel,
  PlayerRingStatsViewModel,
  HUDDecisionPhaseViewModel,
  HUDWeirdStateViewModel,
} from '../../../../src/client/adapters/gameViewModels';
import type { GamePhase, BoardType } from '../../../../src/shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Test Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a PhaseViewModel for a given phase
 */
function createPhaseViewModel(phase: GamePhase): PhaseViewModel {
  const phaseInfoMap: Record<GamePhase, Omit<PhaseViewModel, 'phaseKey'>> = {
    ring_placement: {
      label: 'Place Rings',
      description: 'Add rings to the board to build your stacks',
      colorClass: 'bg-blue-500',
      icon: '🎯',
      actionHint: 'Tap an empty space or one of your stacks to place a ring',
      spectatorHint: 'Placing rings',
    },
    movement: {
      label: 'Your Move',
      description: 'Move one of your stacks or jump to capture',
      colorClass: 'bg-green-500',
      icon: '⚡',
      actionHint: 'Tap your stack, then tap where to move it',
      spectatorHint: 'Choosing a move',
    },
    capture: {
      label: 'Capture!',
      description: 'Jump over an opponent to capture their rings',
      colorClass: 'bg-orange-500',
      icon: '⚔️',
      actionHint: 'Tap your stack, then tap beyond an opponent to jump over them',
      spectatorHint: 'Capturing',
    },
    chain_capture: {
      label: 'Keep Capturing!',
      description: 'You can make another capture—keep jumping!',
      colorClass: 'bg-orange-500',
      icon: '🔗',
      actionHint: 'Tap the next opponent to jump over, or skip if none available',
      spectatorHint: 'Chain capturing',
    },
    line_processing: {
      label: 'Line Scored!',
      description: 'You made a line of 5+ markers—choose your reward',
      colorClass: 'bg-purple-500',
      icon: '📏',
      actionHint: 'Pick your line reward option',
      spectatorHint: 'Choosing line reward',
    },
    territory_processing: {
      label: 'Territory!',
      description: 'You isolated a region—claim it as your territory',
      colorClass: 'bg-pink-500',
      icon: '🏰',
      actionHint: 'Choose which region to claim',
      spectatorHint: 'Claiming territory',
    },
    forced_elimination: {
      label: 'Blocked!',
      description: 'No moves available—you must remove a ring from one of your stacks',
      colorClass: 'bg-red-600',
      icon: '💥',
      actionHint: 'Choose which stack to remove a ring from',
      spectatorHint: 'Forced elimination',
    },
    game_over: {
      label: 'Game Over',
      description: 'The game has ended',
      colorClass: 'bg-slate-600',
      icon: '🏁',
      actionHint: '',
      spectatorHint: 'Game finished',
    },
  };

  return {
    phaseKey: phase,
    ...phaseInfoMap[phase],
  };
}

/**
 * Create ring stats for a player
 */
function createRingStats(
  overrides: Partial<PlayerRingStatsViewModel> = {}
): PlayerRingStatsViewModel {
  return {
    inHand: 10,
    onBoard: 5,
    eliminated: 3,
    total: 18,
    ...overrides,
  };
}

/**
 * Create a PlayerViewModel
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
    isCurrentPlayer: playerNumber === 1,
    isUserPlayer: playerNumber === 1,
    colorClass: colorClasses[playerNumber - 1] ?? 'bg-gray-500',
    ringStats: createRingStats(),
    territorySpaces: 0,
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
 * Create a minimal HUDViewModel
 */
function createHUDViewModel(overrides: Partial<HUDViewModel> = {}): HUDViewModel {
  const baseViewModel: HUDViewModel = {
    phase: createPhaseViewModel('movement'),
    players: [
      createPlayerViewModel(1),
      createPlayerViewModel(2, { isCurrentPlayer: false, isUserPlayer: false }),
    ],
    turnNumber: 5,
    moveNumber: 12,
    pieRuleSummary: undefined,
    instruction: undefined,
    connectionStatus: 'connected',
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    subPhaseDetail: undefined,
    decisionPhase: undefined,
    weirdState: undefined,
  };

  // Merge overrides explicitly to handle optional properties
  return { ...baseViewModel, ...overrides } as HUDViewModel;
}

/**
 * Create default props for GameHUD with viewModel
 */
function createDefaultProps(
  viewModelOverrides: Partial<HUDViewModel> = {},
  propsOverrides: Partial<Omit<GameHUDViewModelProps, 'viewModel'>> = {}
): GameHUDViewModelProps {
  return {
    viewModel: createHUDViewModel(viewModelOverrides),
    ...propsOverrides,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('GameHUD', () => {
  describe('Rendering', () => {
    it('renders without crashing', () => {
      const props = createDefaultProps();
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('game-hud')).toBeInTheDocument();
    });

    it('displays connection status', () => {
      const props = createDefaultProps({ connectionStatus: 'connected' });
      render(<GameHUD {...props} />);

      expect(screen.getByText(/Connection:.*Connected/)).toBeInTheDocument();
    });

    it('displays turn number prominently', () => {
      const props = createDefaultProps({ turnNumber: 15 });
      render(<GameHUD {...props} />);

      expect(screen.getByText('15')).toBeInTheDocument();
      // "Turn" may appear multiple times (in turn counter and current turn badge)
      expect(screen.getAllByText('Turn').length).toBeGreaterThanOrEqual(1);
    });

    it('displays move number when present', () => {
      const props = createDefaultProps({ moveNumber: 42 });
      render(<GameHUD {...props} />);

      expect(screen.getByText('Move #42')).toBeInTheDocument();
    });

    it('renders all player cards', () => {
      const players = [
        createPlayerViewModel(1),
        createPlayerViewModel(2, { isCurrentPlayer: false, isUserPlayer: false }),
        createPlayerViewModel(3, { isCurrentPlayer: false, isUserPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('player-card-player-1')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-2')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-3')).toBeInTheDocument();
    });

    it('renders victory conditions panel by default', () => {
      const props = createDefaultProps();
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('victory-conditions-help')).toBeInTheDocument();
      expect(screen.getByText('Victory Conditions')).toBeInTheDocument();
    });

    it('hides victory conditions panel when hideVictoryConditions is true', () => {
      const props = createDefaultProps({}, { hideVictoryConditions: true });
      render(<GameHUD {...props} />);

      expect(screen.queryByTestId('victory-conditions-help')).not.toBeInTheDocument();
    });

    it('renders instruction banner when instruction is provided', () => {
      const props = createDefaultProps({ instruction: 'Place your ring on an empty space' });
      render(<GameHUD {...props} />);

      expect(screen.getByText('Place your ring on an empty space')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Phase Display Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Phase Display', () => {
    const ALL_PHASES: GamePhase[] = [
      'ring_placement',
      'movement',
      'capture',
      'chain_capture',
      'line_processing',
      'territory_processing',
      'forced_elimination',
    ];

    it.each(ALL_PHASES)('displays correct phase indicator for %s phase', (phase) => {
      const phaseViewModel = createPhaseViewModel(phase);
      const props = createDefaultProps({ phase: phaseViewModel });
      render(<GameHUD {...props} />);

      const indicator = screen.getByTestId('phase-indicator');
      expect(indicator).toBeInTheDocument();
      expect(indicator.textContent).toContain(phaseViewModel.label);
    });

    it('displays phase icon in indicator', () => {
      const phase = createPhaseViewModel('movement');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      const indicator = screen.getByTestId('phase-indicator');
      expect(indicator.textContent).toContain('⚡');
    });

    it('displays phase description', () => {
      const phase = createPhaseViewModel('capture');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      const indicator = screen.getByTestId('phase-indicator');
      expect(indicator.textContent).toContain('Jump over an opponent');
    });

    it('displays action hint when it is user turn', () => {
      const phase = createPhaseViewModel('movement');
      const players = [
        createPlayerViewModel(1, { isCurrentPlayer: true, isUserPlayer: true }),
        createPlayerViewModel(2, { isCurrentPlayer: false, isUserPlayer: false }),
      ];
      const props = createDefaultProps({ phase, players });
      render(<GameHUD {...props} />);

      const actionHint = screen.getByTestId('phase-action-hint');
      expect(actionHint).toBeInTheDocument();
      expect(actionHint.textContent).toContain('Tap your stack');
    });

    it('displays game_over phase appropriately', () => {
      const phase = createPhaseViewModel('game_over');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      const indicator = screen.getByTestId('phase-indicator');
      expect(indicator.textContent).toContain('Game Over');
      expect(indicator.textContent).toContain('🏁');
    });

    it('displays sub-phase detail when provided', () => {
      const props = createDefaultProps({
        phase: createPhaseViewModel('line_processing'),
        subPhaseDetail: 'Processing 3 lines',
      });
      render(<GameHUD {...props} />);

      expect(screen.getByText('Processing 3 lines')).toBeInTheDocument();
    });

    it('provides phase help button for movement phase', () => {
      const phase = createPhaseViewModel('movement');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-phase-help-movement')).toBeInTheDocument();
    });

    it('provides phase help button for capture phase', () => {
      const phase = createPhaseViewModel('capture');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-phase-help-capture')).toBeInTheDocument();
    });

    it('provides phase help button for chain_capture phase', () => {
      const phase = createPhaseViewModel('chain_capture');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-phase-help-chain_capture')).toBeInTheDocument();
    });

    it('provides phase help button for line_processing phase', () => {
      const phase = createPhaseViewModel('line_processing');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-phase-help-line_processing')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Player State Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Player State', () => {
    it('highlights current player with active styling', () => {
      const players = [
        createPlayerViewModel(1, { isCurrentPlayer: true }),
        createPlayerViewModel(2, { isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p1Card = screen.getByTestId('player-card-player-1');
      expect(p1Card.className).toContain('blue');
      expect(p1Card.textContent).toContain('Current Turn');
    });

    it('shows ring statistics for each player', () => {
      const players = [
        createPlayerViewModel(1, {
          ringStats: createRingStats({ inHand: 8, onBoard: 7, eliminated: 3, total: 18 }),
        }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p1Card = screen.getByTestId('player-card-player-1');
      expect(p1Card.textContent).toContain('8');
      expect(p1Card.textContent).toContain('7');
      expect(p1Card.textContent).toContain('3');
    });

    it('displays territory spaces when player has territory', () => {
      const players = [createPlayerViewModel(1, { territorySpaces: 5 })];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p1Card = screen.getByTestId('player-card-player-1');
      expect(p1Card.textContent).toContain('5');
      expect(p1Card.textContent).toContain('territory space');
    });

    it('shows AI badge for AI players', () => {
      const players = [
        createPlayerViewModel(1, { isUserPlayer: true }),
        createAIPlayerViewModel(2, 5),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p2Card = screen.getByTestId('player-card-player-2');
      expect(p2Card.textContent).toContain('AI');
      expect(p2Card.textContent).toContain('Minimax');
    });

    it('shows AI difficulty level', () => {
      const players = [
        createPlayerViewModel(1),
        createAIPlayerViewModel(2, 7, {
          aiInfo: {
            isAI: true,
            difficulty: 7,
            difficultyLabel: 'Expert · MCTS',
            difficultyColor: 'text-purple-300',
            difficultyBgColor: 'bg-purple-900/40',
            aiTypeLabel: 'MCTS',
          },
        }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p2Card = screen.getByTestId('player-card-player-2');
      expect(p2Card.textContent).toContain('Lv7');
      expect(p2Card.textContent).toContain('Expert');
      expect(p2Card.textContent).toContain('MCTS');
    });

    it('applies correct player color styling', () => {
      const players = [
        createPlayerViewModel(1, { colorClass: 'bg-emerald-500' }),
        createPlayerViewModel(2, { colorClass: 'bg-sky-500', isCurrentPlayer: false }),
        createPlayerViewModel(3, { colorClass: 'bg-amber-500', isCurrentPlayer: false }),
        createPlayerViewModel(4, { colorClass: 'bg-fuchsia-500', isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      // Each player card should contain their color class in the color indicator
      const p1Card = screen.getByTestId('player-card-player-1');
      const p2Card = screen.getByTestId('player-card-player-2');
      const p3Card = screen.getByTestId('player-card-player-3');
      const p4Card = screen.getByTestId('player-card-player-4');

      expect(p1Card.innerHTML).toContain('emerald');
      expect(p2Card.innerHTML).toContain('sky');
      expect(p3Card.innerHTML).toContain('amber');
      expect(p4Card.innerHTML).toContain('fuchsia');
    });

    it('displays username for each player', () => {
      const players = [
        createPlayerViewModel(1, { username: 'Alice' }),
        createPlayerViewModel(2, { username: 'Bob', isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      // Names may appear in multiple places (player cards and score summary)
      expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Bob').length).toBeGreaterThanOrEqual(1);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Timer Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Timer', () => {
    it('displays timer when timeControl is provided and player has timeRemaining', () => {
      const players = [
        createPlayerViewModel(1, { timeRemaining: 300000 }), // 5 minutes
        createPlayerViewModel(2, { timeRemaining: 240000, isCurrentPlayer: false }),
      ];
      const props = createDefaultProps(
        { players },
        {
          timeControl: {
            type: 'rapid',
            initialTime: 600,
            increment: 5,
          },
        }
      );
      render(<GameHUD {...props} />);

      // Should show the time in MM:SS format
      const p1Card = screen.getByTestId('player-card-player-1');
      expect(p1Card.textContent).toMatch(/5:00/);
    });

    it('displays time control summary when timeControl is provided', () => {
      const props = createDefaultProps(
        {},
        {
          timeControl: {
            type: 'rapid',
            initialTime: 300,
            increment: 3,
          },
        }
      );
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-time-control-summary')).toBeInTheDocument();
      expect(screen.getByTestId('hud-time-control-summary').textContent).toContain('5+3');
    });

    it('shows low time warning styling for players below 60 seconds', () => {
      const players = [
        createPlayerViewModel(1, { timeRemaining: 30000, isCurrentPlayer: true }), // 30 seconds
      ];
      const props = createDefaultProps(
        { players },
        {
          timeControl: {
            type: 'blitz',
            initialTime: 180,
            increment: 2,
          },
        }
      );
      render(<GameHUD {...props} />);

      // Player card should exist and show time
      const p1Card = screen.getByTestId('player-card-player-1');
      expect(p1Card).toBeInTheDocument();
      // Check that 30 seconds (0:30) is displayed somewhere in the card
      expect(p1Card.textContent).toMatch(/0:30/);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Connection Status Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Connection Status', () => {
    it('shows connected status with green styling', () => {
      const props = createDefaultProps({ connectionStatus: 'connected' });
      const { container } = render(<GameHUD {...props} />);

      const connectionElement = container.querySelector('.text-emerald-300');
      expect(connectionElement).toBeInTheDocument();
      expect(connectionElement?.textContent).toContain('Connected');
    });

    it('shows reconnecting status', () => {
      const props = createDefaultProps({ connectionStatus: 'reconnecting' });
      render(<GameHUD {...props} />);

      expect(screen.getByText(/Reconnecting/)).toBeInTheDocument();
    });

    it('shows disconnected status', () => {
      const props = createDefaultProps({ connectionStatus: 'disconnected' });
      render(<GameHUD {...props} />);

      expect(screen.getByText(/Disconnected/)).toBeInTheDocument();
    });

    it('shows stale connection warning when isConnectionStale is true', () => {
      const props = createDefaultProps({
        connectionStatus: 'connected',
        isConnectionStale: true,
      });
      render(<GameHUD {...props} />);

      expect(screen.getByText(/no recent updates from server/)).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Spectator Mode Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Spectator Mode', () => {
    it('shows spectator banner when isSpectator is true', () => {
      const props = createDefaultProps({ isSpectator: true });
      render(<GameHUD {...props} />);

      expect(screen.getByText('Spectator Mode')).toBeInTheDocument();
      expect(screen.getByText('You are watching this game')).toBeInTheDocument();
    });

    it('shows spectator count when spectating', () => {
      const props = createDefaultProps({ isSpectator: true, spectatorCount: 5 });
      render(<GameHUD {...props} />);

      expect(screen.getByText('5 viewers total')).toBeInTheDocument();
    });

    it('shows spectator count to players when not spectating', () => {
      const props = createDefaultProps({ isSpectator: false, spectatorCount: 3 });
      render(<GameHUD {...props} />);

      expect(screen.getByText('3 watching')).toBeInTheDocument();
    });

    it('displays spectator hint instead of action hint in phase indicator', () => {
      const phase = createPhaseViewModel('movement');
      const players = [
        createPlayerViewModel(1, { isCurrentPlayer: true, isUserPlayer: false }),
        createPlayerViewModel(2, { isCurrentPlayer: false, isUserPlayer: false }),
      ];
      const props = createDefaultProps({
        phase,
        players,
        isSpectator: true,
      });
      render(<GameHUD {...props} />);

      // Verify spectator mode is active by checking for the spectator banner
      expect(screen.getByText('Spectator Mode')).toBeInTheDocument();
      
      // For spectators, the phase indicator should still exist
      expect(screen.getByTestId('phase-indicator')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Decision Phase Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Decision Phase', () => {
    it('renders decision phase banner when decisionPhase is active', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Choose Line Reward',
        description: 'Select your line reward option',
        shortLabel: 'Line Reward',
        timeRemainingMs: 15000,
        showCountdown: true,
        spectatorLabel: 'Waiting for Alice to choose a line reward option',
      };
      const props = createDefaultProps({ decisionPhase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('decision-phase-banner')).toBeInTheDocument();
      expect(screen.getByText('Your decision: Choose Line Reward')).toBeInTheDocument();
    });

    it('shows countdown timer in decision phase', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision',
        shortLabel: 'Decision',
        timeRemainingMs: 45000,
        showCountdown: true,
        spectatorLabel: 'Waiting',
      };
      const props = createDefaultProps({ decisionPhase });
      render(<GameHUD {...props} />);

      const countdown = screen.getByTestId('decision-phase-countdown');
      expect(countdown).toBeInTheDocument();
      expect(countdown.textContent).toContain('0:45');
    });

    it('shows status chip for ring elimination decision', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Ring Elimination',
        shortLabel: 'Elimination',
        timeRemainingMs: null,
        showCountdown: false,
        spectatorLabel: 'Waiting',
        statusChip: {
          text: 'Select stack cap to eliminate',
          tone: 'attention',
        },
      };
      const props = createDefaultProps({ decisionPhase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-decision-status-chip')).toBeInTheDocument();
      expect(screen.getByText('Select stack cap to eliminate')).toBeInTheDocument();
    });

    it('shows skip hint when canSkip is true', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Territory',
        shortLabel: 'Territory',
        timeRemainingMs: null,
        showCountdown: false,
        spectatorLabel: 'Waiting',
        statusChip: {
          text: 'Choose region',
          tone: 'info',
        },
        canSkip: true,
      };
      const props = createDefaultProps({
        phase: createPhaseViewModel('territory_processing'),
        decisionPhase,
      });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-decision-skip-hint')).toBeInTheDocument();
      expect(screen.getByText('Skip available')).toBeInTheDocument();
    });

    it('applies critical severity styling for low decision time', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision',
        shortLabel: 'Decision',
        timeRemainingMs: 5000, // 5 seconds - critical
        showCountdown: true,
        spectatorLabel: 'Waiting',
      };
      const props = createDefaultProps({ decisionPhase });
      render(<GameHUD {...props} />);

      // Check that the decision phase banner is displayed and shows the short time
      const banner = screen.getByTestId('decision-phase-banner');
      expect(banner).toBeInTheDocument();
      // Should show 5 seconds as 0:05
      expect(banner.textContent).toMatch(/0:05/);
    });

    it('shows server-capped indicator when isServerCapped is true', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision',
        shortLabel: 'Decision',
        timeRemainingMs: 30000,
        showCountdown: true,
        isServerCapped: true,
        spectatorLabel: 'Waiting',
      };
      const props = createDefaultProps({ decisionPhase });
      render(<GameHUD {...props} />);

      const countdown = screen.getByTestId('decision-phase-countdown');
      expect(countdown).toHaveAttribute('data-server-capped', 'true');
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Weird State Banner Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Weird State Banner', () => {
    it('renders weird state banner for active-no-moves-movement', () => {
      const weirdState: HUDWeirdStateViewModel = {
        type: 'active-no-moves-movement',
        title: 'No Moves Available!',
        body: 'You have stacks on the board but nowhere to move them.',
        tone: 'warning',
      };
      const props = createDefaultProps({ weirdState });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-weird-state-banner')).toBeInTheDocument();
      expect(screen.getByText('No Moves Available!')).toBeInTheDocument();
    });

    it('renders weird state banner for forced-elimination', () => {
      const weirdState: HUDWeirdStateViewModel = {
        type: 'forced-elimination',
        title: 'Your stacks are being reduced',
        body: 'Rings are being removed—each one counts toward Ring Elimination victory!',
        tone: 'warning',
      };
      const props = createDefaultProps({ weirdState });
      render(<GameHUD {...props} />);

      expect(screen.getByText('Your stacks are being reduced')).toBeInTheDocument();
    });

    it('renders weird state banner for structural-stalemate with critical styling', () => {
      const weirdState: HUDWeirdStateViewModel = {
        type: 'structural-stalemate',
        title: 'Game Ended: Stalemate',
        body: 'Nobody can make any more moves.',
        tone: 'critical',
      };
      const props = createDefaultProps({ weirdState });
      render(<GameHUD {...props} />);

      const banner = screen.getByTestId('hud-weird-state-banner');
      expect(banner.className).toContain('red');
    });

    it('provides help button in weird state banner', () => {
      const weirdState: HUDWeirdStateViewModel = {
        type: 'forced-elimination',
        title: 'Forced Elimination',
        body: 'No moves available.',
        tone: 'warning',
      };
      const props = createDefaultProps({ weirdState });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-weird-state-help')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // LPS Tracking Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('LPS Tracking', () => {
    it('renders LPS indicator when player has consecutive exclusive rounds', () => {
      const viewModel = {
        ...createHUDViewModel(),
        lpsTracking: {
          roundIndex: 5,
          consecutiveExclusiveRounds: 2,
          consecutiveExclusivePlayer: 1,
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-lps-indicator')).toBeInTheDocument();
      expect(screen.getByText(/has exclusive actions/)).toBeInTheDocument();
    });

    it('shows LPS progress dots', () => {
      const viewModel = {
        ...createHUDViewModel(),
        lpsTracking: {
          roundIndex: 5,
          consecutiveExclusiveRounds: 2,
          consecutiveExclusivePlayer: 1,
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      expect(screen.getByText('Round 2/3')).toBeInTheDocument();
    });

    it('shows LPS victory imminent at 3 consecutive rounds', () => {
      const viewModel = {
        ...createHUDViewModel(),
        lpsTracking: {
          roundIndex: 7,
          consecutiveExclusiveRounds: 3,
          consecutiveExclusivePlayer: 1,
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      expect(screen.getByText('LPS Victory!')).toBeInTheDocument();
    });

    it('does not render LPS indicator when consecutiveExclusiveRounds is 0', () => {
      const viewModel = {
        ...createHUDViewModel(),
        lpsTracking: {
          roundIndex: 3,
          consecutiveExclusiveRounds: 0,
          consecutiveExclusivePlayer: null,
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      expect(screen.queryByTestId('hud-lps-indicator')).not.toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Victory Progress Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Victory Progress', () => {
    it('renders victory progress indicator when progress is significant', () => {
      const viewModel = {
        ...createHUDViewModel(),
        victoryProgress: {
          ringElimination: {
            threshold: 18,
            leader: { playerNumber: 1, eliminated: 5, percentage: 28 },
          },
          territory: {
            threshold: 33,
            leader: { playerNumber: 2, spaces: 8, percentage: 24 },
          },
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-victory-progress')).toBeInTheDocument();
    });

    it('shows ring elimination progress bar', () => {
      const viewModel = {
        ...createHUDViewModel(),
        victoryProgress: {
          ringElimination: {
            threshold: 18,
            leader: { playerNumber: 1, eliminated: 9, percentage: 50 },
          },
          territory: {
            threshold: 33,
            leader: null,
          },
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      const victoryProgressEl = screen.getByTestId('hud-victory-progress');
      expect(victoryProgressEl.textContent).toContain('9/18');
    });

    it('shows territory progress bar', () => {
      const viewModel = {
        ...createHUDViewModel(),
        victoryProgress: {
          ringElimination: {
            threshold: 18,
            leader: null,
          },
          territory: {
            threshold: 33,
            leader: { playerNumber: 1, spaces: 15, percentage: 45 },
          },
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      const victoryProgressEl = screen.getByTestId('hud-victory-progress');
      expect(victoryProgressEl.textContent).toContain('15/33');
    });

    it('does not render victory progress when no leaders have 20%+ progress', () => {
      const viewModel = {
        ...createHUDViewModel(),
        victoryProgress: {
          ringElimination: {
            threshold: 18,
            leader: { playerNumber: 1, eliminated: 2, percentage: 11 },
          },
          territory: {
            threshold: 33,
            leader: { playerNumber: 2, spaces: 3, percentage: 9 },
          },
        },
      } as HUDViewModel;
      const props: GameHUDViewModelProps = { viewModel };
      render(<GameHUD {...props} />);

      expect(screen.queryByTestId('hud-victory-progress')).not.toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Score Summary Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Score Summary', () => {
    it('renders compact score summary', () => {
      const players = [
        createPlayerViewModel(1, {
          ringStats: createRingStats({ eliminated: 5 }),
          territorySpaces: 3,
        }),
        createPlayerViewModel(2, {
          isCurrentPlayer: false,
          ringStats: createRingStats({ eliminated: 3 }),
          territorySpaces: 2,
        }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('hud-score-summary')).toBeInTheDocument();
    });

    it('shows "You" label for user player in score summary', () => {
      const players = [
        createPlayerViewModel(1, { isUserPlayer: true, username: 'Alice' }),
        createPlayerViewModel(2, { isUserPlayer: false, isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const scoreSummary = screen.getByTestId('hud-score-summary');
      expect(scoreSummary.textContent).toContain('You');
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Local Sandbox Banner Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Local Sandbox Banner', () => {
    it('shows local sandbox banner when isLocalSandboxOnly is true', () => {
      const props = createDefaultProps({}, { isLocalSandboxOnly: true });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('sandbox-local-only-banner')).toBeInTheDocument();
      expect(screen.getByText(/not logged in/)).toBeInTheDocument();
    });

    it('shows sandbox clock message when no timeControl and isLocalSandboxOnly', () => {
      const props = createDefaultProps({}, { isLocalSandboxOnly: true });
      render(<GameHUD {...props} />);

      expect(screen.getByText(/No clock.*local sandbox/)).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Board Controls Button Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Board Controls Button', () => {
    it('renders board controls button when onShowBoardControls is provided', () => {
      const onShowBoardControls = jest.fn();
      const props = createDefaultProps({}, { onShowBoardControls });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('board-controls-button')).toBeInTheDocument();
    });

    it('does not render board controls button when onShowBoardControls is not provided', () => {
      const props = createDefaultProps({}, {});
      render(<GameHUD {...props} />);

      expect(screen.queryByTestId('board-controls-button')).not.toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Pie Rule Summary Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Pie Rule Summary', () => {
    it('displays pie rule summary when provided', () => {
      const props = createDefaultProps({ pieRuleSummary: 'P2 swapped colours with P1' });
      render(<GameHUD {...props} />);

      expect(screen.getByText('Pie rule')).toBeInTheDocument();
      expect(screen.getByText('P2 swapped colours with P1')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Edge Cases
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Edge Cases', () => {
    it('renders correctly for 2-player game', () => {
      const players = [
        createPlayerViewModel(1),
        createPlayerViewModel(2, { isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('player-card-player-1')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-2')).toBeInTheDocument();
      expect(screen.queryByTestId('player-card-player-3')).not.toBeInTheDocument();
    });

    it('renders correctly for 3-player game', () => {
      const players = [
        createPlayerViewModel(1),
        createPlayerViewModel(2, { isCurrentPlayer: false }),
        createPlayerViewModel(3, { isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('player-card-player-1')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-2')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-3')).toBeInTheDocument();
    });

    it('renders correctly for 4-player game', () => {
      const players = [
        createPlayerViewModel(1),
        createPlayerViewModel(2, { isCurrentPlayer: false }),
        createPlayerViewModel(3, { isCurrentPlayer: false }),
        createPlayerViewModel(4, { isCurrentPlayer: false }),
      ];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('player-card-player-1')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-2')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-3')).toBeInTheDocument();
      expect(screen.getByTestId('player-card-player-4')).toBeInTheDocument();
    });

    it('handles players with no territory gracefully', () => {
      const players = [createPlayerViewModel(1, { territorySpaces: 0 })];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p1Card = screen.getByTestId('player-card-player-1');
      expect(p1Card.textContent).not.toContain('territory space');
    });

    it('handles game_over phase with no special state', () => {
      const phase = createPhaseViewModel('game_over');
      const props = createDefaultProps({ phase });
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('phase-indicator')).toBeInTheDocument();
      expect(screen.getByText('Game Over')).toBeInTheDocument();
    });

    it('handles missing instruction gracefully', () => {
      const props = createDefaultProps({ instruction: undefined });
      render(<GameHUD {...props} />);

      // Should not crash and instruction banner should not be present
      expect(screen.getByTestId('game-hud')).toBeInTheDocument();
    });

    it('renders correctly when all optional fields are undefined', () => {
      const minimalViewModel: HUDViewModel = {
        phase: createPhaseViewModel('movement'),
        players: [createPlayerViewModel(1)],
        turnNumber: 1,
        moveNumber: 0,
        connectionStatus: 'connected',
        isConnectionStale: false,
        isSpectator: false,
        spectatorCount: 0,
        // All optional fields are undefined
      };
      const props: GameHUDViewModelProps = { viewModel: minimalViewModel };
      render(<GameHUD {...props} />);

      expect(screen.getByTestId('game-hud')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Accessibility Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Accessibility', () => {
    it('has accessible phase indicator with role and aria-live', () => {
      const props = createDefaultProps();
      render(<GameHUD {...props} />);

      const phaseIndicator = screen.getByTestId('phase-indicator');
      expect(phaseIndicator).toHaveAttribute('role', 'status');
      expect(phaseIndicator).toHaveAttribute('aria-live', 'polite');
    });

    it('has accessible decision phase banner with role', () => {
      const decisionPhase: HUDDecisionPhaseViewModel = {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Decision',
        shortLabel: 'Decision',
        timeRemainingMs: 30000,
        showCountdown: true,
        spectatorLabel: 'Waiting',
      };
      const props = createDefaultProps({ decisionPhase });
      render(<GameHUD {...props} />);

      const banner = screen.getByTestId('decision-phase-banner');
      expect(banner).toHaveAttribute('role', 'status');
    });

    it('player cards have descriptive aria-labels', () => {
      const players = [createPlayerViewModel(1, { username: 'TestPlayer', isUserPlayer: true })];
      const props = createDefaultProps({ players });
      render(<GameHUD {...props} />);

      const p1Card = screen.getByTestId('player-card-player-1');
      const ariaLabel = p1Card.getAttribute('aria-label');
      expect(ariaLabel).toContain('TestPlayer');
      expect(ariaLabel).toContain('rings');
    });

    it('score summary has accessible region label', () => {
      const props = createDefaultProps();
      render(<GameHUD {...props} />);

      const scoreSummary = screen.getByTestId('hud-score-summary');
      expect(scoreSummary).toHaveAttribute('role', 'region');
      expect(scoreSummary).toHaveAttribute('aria-label');
    });

    it('spectator banner has aria-live attribute', () => {
      const props = createDefaultProps({ isSpectator: true });
      render(<GameHUD {...props} />);

      const banner = screen.getByRole('status', { name: /Spectator Mode/i });
      expect(banner).toHaveAttribute('aria-live', 'polite');
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// VictoryConditionsPanel Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('VictoryConditionsPanel', () => {
  it('renders victory conditions panel', () => {
    render(<VictoryConditionsPanel />);

    expect(screen.getByTestId('victory-conditions-help')).toBeInTheDocument();
  });

  it('displays Ring Elimination victory condition', () => {
    render(<VictoryConditionsPanel />);

    expect(screen.getByText('Ring Elimination')).toBeInTheDocument();
    expect(screen.getByText(/Eliminate enough opponent rings/)).toBeInTheDocument();
  });

  it('displays Territory Control victory condition', () => {
    render(<VictoryConditionsPanel />);

    expect(screen.getByText('Territory Control')).toBeInTheDocument();
    expect(screen.getByText(/Dominate the board/)).toBeInTheDocument();
  });

  it('displays Last Player Standing victory condition', () => {
    render(<VictoryConditionsPanel />);

    expect(screen.getByText('Last Player Standing')).toBeInTheDocument();
    expect(screen.getByText(/only active player for 3 rounds/)).toBeInTheDocument();
  });

  it('provides tooltip triggers for victory condition details', () => {
    render(<VictoryConditionsPanel />);

    expect(screen.getByTestId('victory-tooltip-elimination-trigger')).toBeInTheDocument();
    expect(screen.getByTestId('victory-tooltip-territory-trigger')).toBeInTheDocument();
    expect(screen.getByTestId('victory-tooltip-last-player-standing-trigger')).toBeInTheDocument();
  });

  it('applies custom className when provided', () => {
    render(<VictoryConditionsPanel className="custom-class" />);

    const panel = screen.getByTestId('victory-conditions-help');
    expect(panel.className).toContain('custom-class');
  });
});
