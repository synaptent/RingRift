import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { GameState, Player } from '../../../src/shared/types/game';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';
import { TEACHING_TOPICS_COPY } from '../../../src/client/utils/rulesUxTelemetry';

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
    },
    {
      id: 'p2',
      username: 'Bot',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 90_000,
      ringsInHand: 4,
      eliminatedRings: 2,
      territorySpaces: 0,
      aiDifficulty: 3,
      aiProfile: { difficulty: 3, aiType: 'minimax' },
    },
  ];
}

function createGameState(): GameState {
  const players = createPlayers();
  return {
    id: 'game-1',
    boardType: 'square8',
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
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };
}

describe('GameHUD – legacy props', () => {
  it('renders connection status, spectator badge, ring and territory stats', () => {
    const gameState = createGameState();
    const currentPlayer = gameState.players[0];

    render(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        instruction="Select a stack to move."
        connectionStatus="connected"
        isSpectator={true}
        currentUserId="p1"
      />
    );

    // Connection line and instruction banner
    expect(screen.getByText(/Connection: Connected/i)).toBeInTheDocument();
    expect(screen.getByText('Select a stack to move.')).toBeInTheDocument();

    // Spectator badge
    expect(screen.getByText(/Spectator/i)).toBeInTheDocument();

    // Ring stats labels (Captured was renamed to Rings Eliminated in ARIA update)
    expect(screen.getAllByText('In Hand').length).toBeGreaterThan(0);
    expect(screen.getAllByText('On Board').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Rings Eliminated').length).toBeGreaterThan(0);

    // Territory stats: only first player has non-zero spaces
    // The text is split across elements (<span>2</span> territory spaces)
    // so we use a custom matcher that checks the full text content
    expect(
      screen.getByText((_, element) => element?.textContent === '2 territory spaces')
    ).toBeInTheDocument();
  });
});

describe('GameHUD – view-model props', () => {
  function createHUDViewModel(): HUDViewModel {
    return {
      phase: {
        phaseKey: 'movement',
        label: 'Movement Phase',
        description: 'Move a stack or capture opponent pieces',
        icon: '⚡',
        colorClass: 'bg-green-500',
        actionHint: 'Select your stack, then click a destination to move',
        spectatorHint: 'Player is choosing a move',
      },
      players: [
        {
          id: 'p1',
          username: 'Alice',
          playerNumber: 1,
          colorClass: 'bg-blue-500',
          isCurrentPlayer: true,
          isUserPlayer: true,
          timeRemaining: 120_000,
          ringStats: { inHand: 5, onBoard: 3, eliminated: 1, total: 9 },
          territorySpaces: 2,
          aiInfo: {
            isAI: false,
            difficulty: 0,
            difficultyLabel: '',
            difficultyColor: '',
            difficultyBgColor: '',
            aiTypeLabel: '',
          },
        },
        {
          id: 'p2',
          username: 'Bot',
          playerNumber: 2,
          colorClass: 'bg-red-500',
          isCurrentPlayer: false,
          isUserPlayer: false,
          timeRemaining: 90_000,
          ringStats: { inHand: 4, onBoard: 4, eliminated: 2, total: 10 },
          territorySpaces: 0,
          aiInfo: {
            isAI: true,
            difficulty: 3,
            difficultyLabel: 'Advanced · Minimax',
            difficultyColor: 'text-blue-300',
            difficultyBgColor: 'bg-blue-900/40',
            aiTypeLabel: 'Minimax',
          },
        },
      ],
      turnNumber: 3,
      moveNumber: 7,
      instruction: 'Select a stack to move.',
      connectionStatus: 'connected',
      isConnectionStale: true,
      isSpectator: false,
      spectatorCount: 1,
      subPhaseDetail: 'Sub-phase detail',
      decisionPhase: undefined,
    };
  }

  it('renders connection status, spectator count, phase, players, and timers from view model', () => {
    jest.useFakeTimers();
    const viewModel = createHUDViewModel();

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    // Connection label with stale hint
    expect(screen.getByText(/Connection: Connected/i)).toBeInTheDocument();
    expect(screen.getByText(/\(no recent updates from server\)/i)).toBeInTheDocument();

    // Spectator count badge (there may be multiple '1's like timer "1:30")
    // Just verify at least one '1' exists; SVG eye icon accompanies spectator count
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);

    // Phase label and description
    expect(screen.getByText(/Movement Phase/i)).toBeInTheDocument();
    expect(screen.getByText(/Move a stack or capture opponent pieces/i)).toBeInTheDocument();

    // Instruction banner
    expect(screen.getByText('Select a stack to move.')).toBeInTheDocument();

    // Player cards: current player + AI badge + territory spaces
    expect(screen.getByText('Alice')).toBeInTheDocument();
    // Multiple Bot labels may appear (player card + score summary), so use a plural query.
    expect(screen.getAllByText(/Bot/).length).toBeGreaterThan(0);
    // Territory spaces text is split across elements, verify at least one matches
    expect(
      screen.getAllByText(
        (_, element) => element?.textContent?.includes('territory space') ?? false
      ).length
    ).toBeGreaterThan(0);
    expect(screen.getByText(/AI/)).toBeInTheDocument();
    expect(screen.getByText(/Advanced · Minimax Lv3/)).toBeInTheDocument();
    // "Minimax" appears in both difficulty badge and AI type label
    expect(screen.getAllByText(/Minimax/).length).toBeGreaterThan(0);

    // Timers render in mm:ss format
    expect(screen.getAllByText(/2:00|1:30/).length).toBeGreaterThan(0);

    jest.useRealTimers();
  });

  it('renders Line Formation phase and ring-elimination status chip when provided', () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      phase: {
        ...baseVm.phase,
        label: 'Line Formation',
        description: 'Resolve completed lines and select any elimination rewards.',
        icon: '✨',
        colorClass: 'bg-amber-500',
      },
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Ring elimination',
        description: 'Choose which stack to eliminate from.',
        shortLabel: 'Ring elimination',
        timeRemainingMs: null,
        showCountdown: false,
        warningThresholdMs: undefined,
        isServerCapped: undefined,
        spectatorLabel: 'Waiting for Alice to choose a stack for ring elimination',
        statusChip: {
          text: 'Select stack cap to eliminate',
          tone: 'attention',
        },
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    expect(screen.getByText(/Line Formation/i)).toBeInTheDocument();
    const chip = screen.getByTestId('hud-decision-status-chip');
    expect(chip).toHaveTextContent('Select stack cap to eliminate');
    expect(chip).toHaveClass('bg-amber-500');
  });

  it('renders reconnection status with warning styling and no stale hint', () => {
    const viewModel: HUDViewModel = {
      ...createHUDViewModel(),
      connectionStatus: 'reconnecting',
      isConnectionStale: false,
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const connection = screen.getByText(/Connection: Reconnecting/i);
    expect(connection).toBeInTheDocument();
    expect(connection).toHaveClass('text-amber-300');
    expect(screen.queryByText(/\(no recent updates from server\)/i)).toBeNull();
  });

  it('shows decision time-pressure badge with warning severity for local actor', () => {
    const viewModel: HUDViewModel = {
      ...createHUDViewModel(),
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Capture direction',
        description: 'Choose direction',
        shortLabel: 'Capture direction',
        timeRemainingMs: 9_000,
        showCountdown: true,
        warningThresholdMs: 10_000,
        isServerCapped: false,
        spectatorLabel: 'Waiting for Alice',
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const badge = screen.getByTestId('hud-decision-time-pressure');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveAttribute('data-severity', 'warning');
    expect(badge).toHaveTextContent('Your decision timer:');
    expect(badge).toHaveTextContent('0:09');
  });

  it('renders skip hint badge when decisionPhase.canSkip is true', () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Territory region order',
        description: 'Choose a disconnected region to process or skip territory processing.',
        shortLabel: 'Territory processing',
        timeRemainingMs: null,
        showCountdown: false,
        warningThresholdMs: undefined,
        isServerCapped: undefined,
        spectatorLabel: 'Waiting for Alice to choose a territory region',
        statusChip: {
          text: 'Territory claimed – choose region to process or skip',
          tone: 'attention',
        },
        canSkip: true,
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    expect(screen.getByTestId('hud-decision-status-chip')).toBeInTheDocument();
    const skipHint = screen.getByTestId('hud-decision-skip-hint');
    expect(skipHint).toBeInTheDocument();
    expect(skipHint).toHaveTextContent(/Skip available/i);
  });

  it('renders time control summary and compact score summary from view model', () => {
    const viewModel = createHUDViewModel();

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const clockSummary = screen.getByTestId('hud-time-control-summary');
    expect(clockSummary).toBeInTheDocument();
    expect(clockSummary).toHaveTextContent('Rapid • 10+0');

    const scoreSummary = screen.getByTestId('hud-score-summary');
    expect(scoreSummary).toBeInTheDocument();
    expect(scoreSummary).toHaveTextContent(/Score summary/i);
    // The summary should surface both rings and territory information for players.
    expect(scoreSummary).toHaveTextContent(/Rings/i);
    expect(scoreSummary).toHaveTextContent(/Territory/i);
  });

  it('shows sandbox no-clock label when flagged and no time control is provided', () => {
    const viewModel = createHUDViewModel();

    render(<GameHUD viewModel={viewModel} isLocalSandboxOnly={true} />);

    expect(screen.getByText(/Clock: No clock \(local sandbox\)/i)).toBeInTheDocument();
  });

  it('renders a weird-state banner when HUDViewModel.weirdState is provided', () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      weirdState: {
        type: 'forced-elimination',
        title: 'Forced elimination is shrinking your stacks',
        body: 'Because you control stacks but have no legal real moves on some of your turns (no placements, movements, or captures), forced elimination repeatedly removes caps from your stacks.',
        tone: 'warning',
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const banner = screen.getByTestId('hud-weird-state-banner');
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/Forced elimination is shrinking your stacks/i);
    expect(banner).toHaveTextContent(/no legal real moves/i);
  });

  it('does not render a weird-state banner when HUDViewModel.weirdState is undefined', () => {
    const viewModel = createHUDViewModel();

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    expect(screen.queryByTestId('hud-weird-state-banner')).toBeNull();
  });

  it('opens TeachingOverlay for Forced Elimination when clicking the weird-state help button', async () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      weirdState: {
        type: 'forced-elimination',
        title: 'Forced Elimination',
        body: 'You control stacks but have no legal moves; caps will be removed automatically.',
        tone: 'warning',
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const helpButton = screen.getByTestId('hud-weird-state-help');
    fireEvent.click(helpButton);

    // Wait for the TeachingOverlay to render after state update
    await waitFor(() => {
      expect(screen.getByText(TEACHING_TOPICS_COPY.forced_elimination.heading)).toBeInTheDocument();
    });
  });

  it('shows a territory help button during territory_processing decisions and opens TeachingOverlay', async () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      phase: {
        ...baseVm.phase,
        phaseKey: 'territory_processing' as any,
        label: 'Territory Processing',
        description: 'Resolve disconnected regions and territory.',
      },
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Choose region order',
        description: 'Choose which disconnected region to process first.',
        shortLabel: 'Territory region order',
        timeRemainingMs: null,
        showCountdown: false,
        warningThresholdMs: undefined,
        isServerCapped: undefined,
        spectatorLabel: 'Waiting for Alice to choose a region to process first',
        statusChip: {
          text: 'Territory claimed – choose region to process or skip',
          tone: 'attention',
        },
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const helpButton = screen.getByTestId('hud-territory-help');
    expect(helpButton).toBeInTheDocument();

    fireEvent.click(helpButton);

    await waitFor(() => {
      expect(
        screen.getByRole('heading', { name: TEACHING_TOPICS_COPY.territory.heading })
      ).toBeInTheDocument();
    });
  });

  it('shows a line-processing phase help chip and opens TeachingOverlay', async () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      phase: {
        ...baseVm.phase,
        phaseKey: 'line_processing' as any,
        label: 'Line Formation',
        description: 'Resolve completed lines and rewards.',
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const helpButton = screen.getByTestId('hud-phase-help-line_processing');
    expect(helpButton).toBeInTheDocument();

    fireEvent.click(helpButton);

    await waitFor(() => {
      expect(screen.getByText(TEACHING_TOPICS_COPY.line_bonus.heading)).toBeInTheDocument();
    });
  });

  it('shows a chain-capture phase help chip and opens TeachingOverlay', async () => {
    const baseVm = createHUDViewModel();
    const viewModel: HUDViewModel = {
      ...baseVm,
      phase: {
        ...baseVm.phase,
        phaseKey: 'chain_capture' as any,
        label: 'Chain Capture',
        description: 'Continue your capture sequence.',
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    const helpButton = screen.getByTestId('hud-phase-help-chain_capture');
    expect(helpButton).toBeInTheDocument();

    fireEvent.click(helpButton);

    await waitFor(() => {
      expect(
        screen.getByRole('heading', { name: TEACHING_TOPICS_COPY.chain_capture.heading })
      ).toBeInTheDocument();
    });
  });
});
