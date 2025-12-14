import React from 'react';
import { render, screen, getAllByTestId } from '@testing-library/react';
import '@testing-library/jest-dom';

import { GameHUD } from '../../src/client/components/GameHUD';
import {
  toHUDViewModel,
  type HUDViewModel,
  type ToHUDViewModelOptions,
} from '../../src/client/adapters/gameViewModels';
import { createTestGameState, createTestPlayer } from '../utils/fixtures';
import type { GameState, GamePhase, PlayerChoice, Position } from '../../src/shared/types/game';

/**
 * HUD Backend vs Sandbox Equivalence Tests
 *
 * These tests focus on ensuring that, for a given canonical GameState and
 * decision configuration, the adapter-based HUD view model (`HUDViewModel`)
 * and the rendered `GameHUD` surface equivalent semantics regardless of
 * whether the calling site is conceptually a "backend" host
 * (BackendGameHost) or a "sandbox" host (SandboxGameHost).
 *
 * We intentionally model the two hosts via different ToHUDViewModelOptions
 * shapes (connection/heartbeat semantics, spectator flags, etc.) while
 * keeping the underlying GameState + decision identical. For fields that
 * represent core HUD semantics (phase labels, decision copy, countdown
 * severity), the resulting view models — and rendered HUD chips — should
 * match.
 *
 * Intentional differences between backend and sandbox hosts:
 * - Connection semantics: backend has real WebSocket state, sandbox is always "connected"
 * - Spectators: backend supports true spectators, sandbox has no spectators (isSpectator always false)
 * - Heartbeat staleness: backend tracks lastHeartbeatAt, sandbox uses null
 */

// ═══════════════════════════════════════════════════════════════════════════
// Helper fixtures for different phases and decision types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create state for line_processing phase with line_reward_option decision.
 */
function createLineRewardDecisionState(): { gameState: GameState; pendingChoice: PlayerChoice } {
  const player1 = createTestPlayer(1, { id: 'user-1', username: 'Alice' });
  const player2 = createTestPlayer(2, { id: 'user-2', username: 'Bob' });

  const gameState = createTestGameState({
    players: [player1, player2],
    currentPlayer: 1,
    currentPhase: 'line_processing',
    gameStatus: 'active',
  });

  const pendingChoice: PlayerChoice = {
    id: 'choice-equivalence',
    type: 'line_reward_option',
    gameId: gameState.id,
    playerNumber: 1,
    prompt: 'Choose line reward',
    // Concrete options are not inspected by the HUD adapter; they are only
    // used by choiceViewModels to derive copy/semantics.
    options: ['add_ring', 'add_stack'] as any,
    timeoutMs: 4000,
  } as any;

  return { gameState, pendingChoice };
}

// Backward-compatible alias for existing tests
const createCanonicalDecisionState = createLineRewardDecisionState;

/**
 * Create state for territory_processing phase with a region_order decision
 * that includes both a concrete region option and an explicit skip option.
 */
function createTerritoryRegionOrderDecisionState(): {
  gameState: GameState;
  pendingChoice: PlayerChoice;
} {
  const player1 = createTestPlayer(1, { id: 'user-1', username: 'Alice' });
  const player2 = createTestPlayer(2, { id: 'user-2', username: 'Bob' });

  const gameState = createTestGameState({
    players: [player1, player2],
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    gameStatus: 'active',
  });

  const pendingChoice: PlayerChoice = {
    id: 'choice-region-skip',
    type: 'region_order',
    gameId: gameState.id,
    playerNumber: 1,
    prompt: 'Choose region or skip',
    options: [
      {
        regionId: 'region-1',
        size: 4,
        representativePosition: { x: 1, y: 1 } as Position,
        moveId: 'process-region-1',
      },
      {
        regionId: 'skip',
        size: 0,
        representativePosition: { x: 0, y: 0 } as Position,
        moveId: 'skip-territory-processing',
      },
    ] as any,
  } as any;

  return { gameState, pendingChoice };
}

/**
 * Create state for movement phase (no pending decision).
 */
function createMovementPhaseState(): { gameState: GameState } {
  const player1 = createTestPlayer(1, { id: 'user-1', username: 'Alice' });
  const player2 = createTestPlayer(2, { id: 'user-2', username: 'Bob' });

  const gameState = createTestGameState({
    players: [player1, player2],
    currentPlayer: 1,
    currentPhase: 'movement',
    gameStatus: 'active',
  });

  return { gameState };
}

/**
 * Create state for chain_capture phase with capture_direction decision.
 */
function createCaptureDirectionDecisionState(): {
  gameState: GameState;
  pendingChoice: PlayerChoice;
} {
  const player1 = createTestPlayer(1, { id: 'user-1', username: 'Alice' });
  const player2 = createTestPlayer(2, { id: 'user-2', username: 'Bob' });

  const gameState = createTestGameState({
    players: [player1, player2],
    currentPlayer: 1,
    currentPhase: 'chain_capture',
    gameStatus: 'active',
  });

  const pendingChoice: PlayerChoice = {
    id: 'choice-capture-dir',
    type: 'capture_direction',
    gameId: gameState.id,
    playerNumber: 1,
    prompt: 'Choose capture direction',
    options: [
      {
        targetPosition: { x: 2, y: 2 },
        landingPosition: { x: 3, y: 3 },
        moveId: 'move-1',
      },
      {
        targetPosition: { x: 2, y: 0 },
        landingPosition: { x: 3, y: 0 },
        moveId: 'move-2',
      },
    ],
    timeoutMs: 8000,
  } as any;

  return { gameState, pendingChoice };
}

/**
 * Create state for territory_processing phase with region_order decision.
 */
function createRegionOrderDecisionState(): {
  gameState: GameState;
  pendingChoice: PlayerChoice;
} {
  const player1 = createTestPlayer(1, { id: 'user-1', username: 'Alice' });
  const player2 = createTestPlayer(2, { id: 'user-2', username: 'Bob' });

  const gameState = createTestGameState({
    players: [player1, player2],
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    gameStatus: 'active',
  });

  const pendingChoice: PlayerChoice = {
    id: 'choice-region-order',
    type: 'region_order',
    gameId: gameState.id,
    playerNumber: 1,
    prompt: 'Choose region to process',
    options: [
      { regionId: 'region-a', size: 3, representativePosition: { x: 1, y: 1 } },
      { regionId: 'region-b', size: 2, representativePosition: { x: 5, y: 5 } },
      { regionId: 'skip', size: 0, representativePosition: { x: -1, y: -1 } },
    ],
    timeoutMs: 10000,
  } as any;

  return { gameState, pendingChoice };
}

/**
 * Create state for line_processing phase with ring_elimination decision.
 */
function createRingEliminationDecisionState(): {
  gameState: GameState;
  pendingChoice: PlayerChoice;
} {
  const player1 = createTestPlayer(1, { id: 'user-1', username: 'Alice' });
  const player2 = createTestPlayer(2, { id: 'user-2', username: 'Bob' });

  const gameState = createTestGameState({
    players: [player1, player2],
    currentPlayer: 1,
    currentPhase: 'line_processing',
    gameStatus: 'active',
  });

  const pendingChoice: PlayerChoice = {
    id: 'choice-ring-elim',
    type: 'ring_elimination',
    gameId: gameState.id,
    playerNumber: 1,
    prompt: 'Choose stack to eliminate from',
    options: [
      { stackPosition: { x: 3, y: 3 }, eliminationCount: 2, moveId: 'elim-1' },
      { stackPosition: { x: 4, y: 4 }, eliminationCount: 1, moveId: 'elim-2' },
    ],
    timeoutMs: 6000,
  } as any;

  return { gameState, pendingChoice };
}

function createBackendHUDOptions(
  overrides: Partial<ToHUDViewModelOptions> = {}
): ToHUDViewModelOptions {
  return {
    connectionStatus: 'connected',
    lastHeartbeatAt: Date.now(),
    isSpectator: false,
    ...overrides,
  };
}

function createSandboxHUDOptions(
  overrides: Partial<ToHUDViewModelOptions> = {}
): ToHUDViewModelOptions {
  return {
    // Sandbox host treats the HUD as always-connected and does not surface
    // heartbeat staleness semantics.
    connectionStatus: 'connected',
    lastHeartbeatAt: null,
    isSpectator: false,
    ...overrides,
  };
}

function pickCoreDecisionFields(vm: HUDViewModel | null | undefined) {
  if (!vm || !vm.decisionPhase) return null;

  const d = vm.decisionPhase;
  return {
    actingPlayerNumber: d.actingPlayerNumber,
    actingPlayerName: d.actingPlayerName,
    isLocalActor: d.isLocalActor,
    label: d.label,
    shortLabel: d.shortLabel,
    description: d.description,
    spectatorLabel: d.spectatorLabel,
    showCountdown: d.showCountdown,
    timeRemainingMs: d.timeRemainingMs,
    warningThresholdMs: d.warningThresholdMs,
    statusChip: d.statusChip,
    canSkip: d.canSkip,
  } as const;
}

describe('HUD backend vs sandbox equivalence', () => {
  it('produces equivalent phase and decision semantics for backend-style and sandbox-style HUDViewModels', () => {
    const { gameState, pendingChoice } = createCanonicalDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        // Model the already-reconciled countdown as seen by BackendGameHost
        // after useDecisionCountdown has run.
        choiceTimeRemainingMs: 4000,
        choiceDeadline: Date.now() + 4000,
        decisionIsServerCapped: false,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        // Sandbox host derives its countdown directly from timeoutMs; for
        // equivalence we model the same effective remaining time.
        choiceTimeRemainingMs: 4000,
        choiceDeadline: null,
      })
    );

    // Phase semantics (label, description, celebratory "Line Formation" styling)
    expect(backendHud.phase.phaseKey).toBe('line_processing');
    expect(sandboxHud.phase.phaseKey).toBe('line_processing');
    expect(backendHud.phase.label).toBe('Line Formation');
    expect(sandboxHud.phase.label).toBe('Line Formation');
    expect(backendHud.phase.description).toBe(sandboxHud.phase.description);
    expect(backendHud.phase.icon).toBe(sandboxHud.phase.icon);
    expect(backendHud.phase.colorClass).toBe(sandboxHud.phase.colorClass);

    // Core decision-phase semantics should be identical for the same
    // underlying GameState + PlayerChoice when the options contract is
    // respected.
    const backendDecision = pickCoreDecisionFields(backendHud);
    const sandboxDecision = pickCoreDecisionFields(sandboxHud);

    expect(backendDecision).not.toBeNull();
    expect(sandboxDecision).not.toBeNull();
    expect(backendDecision).toEqual(sandboxDecision);
  });

  it('renders the same HUD-level decision time-pressure chip for backend and sandbox HUDs given the same view-model semantics', () => {
    const { gameState, pendingChoice } = createCanonicalDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 4000,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 4000,
      })
    );

    const { getAllByTestId } = render(
      <>
        <div data-testid="backend-hud">
          <GameHUD viewModel={backendHud} timeControl={gameState.timeControl} />
        </div>
        <div data-testid="sandbox-hud">
          <GameHUD viewModel={sandboxHud} timeControl={gameState.timeControl} />
        </div>
      </>
    );

    const chips = getAllByTestId('hud-decision-time-pressure');
    expect(chips).toHaveLength(2);

    const [backendChip, sandboxChip] = chips;

    // Copy and severity should be identical when the underlying
    // decisionPhase semantics match.
    expect(backendChip).toHaveAttribute('data-severity', 'warning');
    expect(sandboxChip).toHaveAttribute('data-severity', 'warning');
    expect(backendChip.textContent).toBe(sandboxChip.textContent);
  });

  it('uses identical spectator-oriented decision copy for backend-style and sandbox-style HUDs when viewing as a pure spectator', () => {
    const { gameState, pendingChoice } = createCanonicalDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        isSpectator: true,
        currentUserId: undefined,
        pendingChoice,
        choiceTimeRemainingMs: null,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        isSpectator: true,
        currentUserId: undefined,
        pendingChoice,
        choiceTimeRemainingMs: null,
      })
    );

    const backendDecision = pickCoreDecisionFields(backendHud);
    const sandboxDecision = pickCoreDecisionFields(sandboxHud);

    expect(backendDecision).not.toBeNull();
    expect(sandboxDecision).not.toBeNull();

    // For pure spectators, the HUD should always use spectator-oriented
    // copy that mentions the acting player's name.
    expect(backendDecision?.isLocalActor).toBe(false);
    expect(sandboxDecision?.isLocalActor).toBe(false);
    // Beginner-friendly spectator copy from choiceViewModels
    expect(backendDecision?.label).toBe('Alice is choosing their line reward');
    expect(sandboxDecision?.label).toBe('Alice is choosing their line reward');
    expect(backendDecision?.spectatorLabel).toBe(sandboxDecision?.spectatorLabel);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Additional phase and decision type equivalence tests
// ═══════════════════════════════════════════════════════════════════════════

describe('HUD equivalence – movement phase (no decision)', () => {
  it('produces equivalent phase semantics for movement phase without pending decision', () => {
    const { gameState } = createMovementPhaseState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({ currentUserId: 'user-1' })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({ currentUserId: 'user-1' })
    );

    // Phase semantics should match (beginner-friendly labels from gameViewModels)
    expect(backendHud.phase.phaseKey).toBe('movement');
    expect(sandboxHud.phase.phaseKey).toBe('movement');
    expect(backendHud.phase.label).toBe('Your Move');
    expect(sandboxHud.phase.label).toBe('Your Move');
    expect(backendHud.phase.description).toBe(sandboxHud.phase.description);
    expect(backendHud.phase.actionHint).toBe(sandboxHud.phase.actionHint);
    expect(backendHud.phase.spectatorHint).toBe(sandboxHud.phase.spectatorHint);

    // No decision phase when no pending choice
    expect(backendHud.decisionPhase).toBeUndefined();
    expect(sandboxHud.decisionPhase).toBeUndefined();
  });
});

describe('HUD equivalence – chain_capture phase with capture_direction decision', () => {
  it('produces equivalent decision semantics for capture_direction choice', () => {
    const { gameState, pendingChoice } = createCaptureDirectionDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 8000,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 8000,
      })
    );

    // Phase semantics
    expect(backendHud.phase.phaseKey).toBe('chain_capture');
    expect(sandboxHud.phase.phaseKey).toBe('chain_capture');
    expect(backendHud.phase.label).toBe(sandboxHud.phase.label);

    // Decision semantics should match
    const backendDecision = pickCoreDecisionFields(backendHud);
    const sandboxDecision = pickCoreDecisionFields(sandboxHud);

    expect(backendDecision).not.toBeNull();
    expect(sandboxDecision).not.toBeNull();
    expect(backendDecision).toEqual(sandboxDecision);

    // Verify label indicates capture direction
    expect(backendDecision?.shortLabel).toContain('Capture');
  });

  it('renders equivalent HUD for capture_direction as spectator', () => {
    const { gameState, pendingChoice } = createCaptureDirectionDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        isSpectator: true,
        currentUserId: undefined,
        pendingChoice,
        choiceTimeRemainingMs: null,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        isSpectator: true,
        currentUserId: undefined,
        pendingChoice,
        choiceTimeRemainingMs: null,
      })
    );

    const backendDecision = pickCoreDecisionFields(backendHud);
    const sandboxDecision = pickCoreDecisionFields(sandboxHud);

    expect(backendDecision?.isLocalActor).toBe(false);
    expect(sandboxDecision?.isLocalActor).toBe(false);
    expect(backendDecision?.spectatorLabel).toBe(sandboxDecision?.spectatorLabel);
  });
});

describe('HUD equivalence – territory_processing phase with region_order decision', () => {
  it('produces equivalent decision semantics for region_order choice', () => {
    const { gameState, pendingChoice } = createRegionOrderDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 10000,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 10000,
      })
    );

    // Phase semantics
    expect(backendHud.phase.phaseKey).toBe('territory_processing');
    expect(sandboxHud.phase.phaseKey).toBe('territory_processing');

    // Decision semantics
    const backendDecision = pickCoreDecisionFields(backendHud);
    const sandboxDecision = pickCoreDecisionFields(sandboxHud);

    expect(backendDecision).not.toBeNull();
    expect(sandboxDecision).not.toBeNull();
    expect(backendDecision).toEqual(sandboxDecision);

    // Region order choices with a skip option should set canSkip
    expect(backendDecision?.canSkip).toBe(true);
    expect(sandboxDecision?.canSkip).toBe(true);

    // Should have a status chip for territory decisions
    expect(backendDecision?.statusChip).toBeDefined();
    expect(sandboxDecision?.statusChip).toBeDefined();
    expect(backendDecision?.statusChip?.tone).toBe('attention');
    expect(sandboxDecision?.statusChip?.tone).toBe('attention');
    expect(backendDecision?.statusChip?.text).toBe(
      'Territory claimed – choose region to process or skip'
    );
    expect(sandboxDecision?.statusChip?.text).toBe(
      'Territory claimed – choose region to process or skip'
    );
  });

  it('renders identical territory region-order status chips and skip hints for backend and sandbox HUDs', () => {
    const { gameState, pendingChoice } = createRegionOrderDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: null,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: null,
      })
    );

    const { getAllByTestId } = render(
      <>
        <div data-testid="backend-hud">
          <GameHUD viewModel={backendHud} timeControl={gameState.timeControl} />
        </div>
        <div data-testid="sandbox-hud">
          <GameHUD viewModel={sandboxHud} timeControl={gameState.timeControl} />
        </div>
      </>
    );

    const chips = getAllByTestId('hud-decision-status-chip');
    expect(chips).toHaveLength(2);

    const [backendChip, sandboxChip] = chips;
    expect(backendChip.textContent).toBe('Territory claimed – choose region to process or skip');
    expect(sandboxChip.textContent).toBe('Territory claimed – choose region to process or skip');

    const skipHints = getAllByTestId('hud-decision-skip-hint');
    expect(skipHints).toHaveLength(2);
    expect(skipHints[0].textContent).toMatch(/Skip available/i);
    expect(skipHints[1].textContent).toMatch(/Skip available/i);
  });
});

describe('HUD equivalence – line_processing phase with ring_elimination decision', () => {
  it('produces equivalent decision semantics for ring_elimination choice', () => {
    const { gameState, pendingChoice } = createRingEliminationDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 6000,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 6000,
      })
    );

    // Phase should show celebratory "Line Formation" styling
    expect(backendHud.phase.label).toBe('Line Formation');
    expect(sandboxHud.phase.label).toBe('Line Formation');
    expect(backendHud.phase.icon).toBe('✨');
    expect(sandboxHud.phase.icon).toBe('✨');

    // Decision semantics
    const backendDecision = pickCoreDecisionFields(backendHud);
    const sandboxDecision = pickCoreDecisionFields(sandboxHud);

    expect(backendDecision).not.toBeNull();
    expect(sandboxDecision).not.toBeNull();
    expect(backendDecision).toEqual(sandboxDecision);

    // Ring elimination should have a status chip with attention tone
    expect(backendDecision?.statusChip).toBeDefined();
    expect(backendDecision?.statusChip?.tone).toBe('attention');
    expect(backendDecision?.statusChip?.text).toContain('eliminate');
  });

  it('renders equivalent time-pressure chip for ring_elimination at warning threshold', () => {
    const { gameState, pendingChoice } = createRingEliminationDecisionState();

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 3500, // In warning range
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        pendingChoice,
        choiceTimeRemainingMs: 3500,
      })
    );

    const { getAllByTestId } = render(
      <>
        <div data-testid="backend-hud">
          <GameHUD viewModel={backendHud} timeControl={gameState.timeControl} />
        </div>
        <div data-testid="sandbox-hud">
          <GameHUD viewModel={sandboxHud} timeControl={gameState.timeControl} />
        </div>
      </>
    );

    const chips = getAllByTestId('hud-decision-time-pressure');
    expect(chips).toHaveLength(2);

    const [backendChip, sandboxChip] = chips;
    // Both should have the same severity for the same remaining time
    expect(backendChip.getAttribute('data-severity')).toBe(
      sandboxChip.getAttribute('data-severity')
    );
  });
});

describe('HUD equivalence – intentional differences documentation', () => {
  it('documents intentional difference: backend tracks heartbeat staleness, sandbox does not', () => {
    const { gameState } = createMovementPhaseState();

    const staleHeartbeat = Date.now() - 15000; // 15s ago, past threshold

    const backendHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        currentUserId: 'user-1',
        lastHeartbeatAt: staleHeartbeat,
      })
    );

    const sandboxHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        currentUserId: 'user-1',
        // Sandbox always passes null for lastHeartbeatAt
      })
    );

    // Backend detects stale connection
    expect(backendHud.isConnectionStale).toBe(true);
    // Sandbox never has stale connection (no heartbeat tracking)
    expect(sandboxHud.isConnectionStale).toBe(false);

    // This is an INTENTIONAL difference – not a bug
    // Backend uses real WebSocket heartbeats; sandbox has no network layer
  });

  it('documents intentional difference: backend supports spectators, sandbox isSpectator is always false in practice', () => {
    const { gameState, pendingChoice } = createLineRewardDecisionState();

    // Backend can have real spectators
    const backendSpectatorHud = toHUDViewModel(
      gameState,
      createBackendHUDOptions({
        isSpectator: true,
        currentUserId: 'spectator-user',
        pendingChoice,
      })
    );

    // Sandbox in normal usage always passes isSpectator: false
    // (there's no network concept of "spectating" a local game)
    const sandboxPlayerHud = toHUDViewModel(
      gameState,
      createSandboxHUDOptions({
        isSpectator: false,
        currentUserId: 'user-1',
        pendingChoice,
      })
    );

    expect(backendSpectatorHud.isSpectator).toBe(true);
    expect(sandboxPlayerHud.isSpectator).toBe(false);

    // This is an INTENTIONAL difference – sandbox has no spectator concept
  });
});
