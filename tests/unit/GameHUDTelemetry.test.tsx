import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../src/client/components/GameHUD';
import type { HUDViewModel } from '../../src/client/adapters/gameViewModels';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';

jest.mock('../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../src/client/utils/rulesUxTelemetry');
  return {
    __esModule: true,
    ...actual,
    // Override with Jest mocks while keeping the rest of the module intact.
    sendRulesUxEvent: jest.fn(),
    logRulesUxEvent: jest.fn(),
    // Stable overlay session id so we can assert linkage between banner and details events.
    newOverlaySessionId: jest.fn(() => 'overlay-session-hud-1'),
  };
});

// Cached typed handles to the mocked telemetry helpers for all tests.
const mockSendRulesUxEvent = rulesUxTelemetry.sendRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.sendRulesUxEvent
>;
const mockLogRulesUxEvent = rulesUxTelemetry.logRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logRulesUxEvent
>;

function createHudWithWeirdState(): HUDViewModel {
  return {
    phase: {
      phaseKey: 'movement' as any,
      label: 'Movement',
      description: 'Move a stack',
      icon: '⚡',
      colorClass: 'bg-green-500',
      actionHint: 'Select a stack then a destination',
      spectatorHint: 'Player is choosing a move',
    },
    players: [],
    turnNumber: 1,
    moveNumber: 0,
    connectionStatus: 'connected' as any,
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    subPhaseDetail: undefined,
    decisionPhase: undefined,
    weirdState: {
      type: 'forced-elimination',
      tone: 'warning',
      title: 'Forced Elimination',
      body: 'You have no legal moves; caps will be removed automatically.',
    } as any,
    pieRuleSummary: undefined,
    instruction: 'Instruction',
  };
}

describe('GameHUD rules-UX telemetry wiring', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('emits legacy and spec-compliant events when weird-state help is opened multiple times', () => {
    const hud = createHudWithWeirdState();

    render(
      <GameHUD
        viewModel={hud}
        timeControl={undefined}
        rulesUxContext={{
          boardType: 'square8' as any,
          numPlayers: 2,
          aiDifficulty: 5,
        }}
      />
    );

    const helpButton = screen.getByTestId('hud-weird-state-help');

    // First click: should show TeachingOverlay and emit weird_state_help + help_open.
    fireEvent.click(helpButton);

    expect(mockSendRulesUxEvent).toHaveBeenCalled();
    expect(
      mockSendRulesUxEvent.mock.calls.some(
        ([arg]: any) =>
          arg.type === 'rules_weird_state_help' &&
          arg.boardType === 'square8' &&
          arg.numPlayers === 2
      )
    ).toBe(true);
    expect(
      mockSendRulesUxEvent.mock.calls.some(
        ([arg]: any) => arg.type === 'rules_help_open' && arg.topic === 'forced_elimination'
      )
    ).toBe(true);

    // Close the overlay via its close button so that a subsequent click counts as a
    // repeat open for the same topic.
    const closeButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeButton);

    // Second click: should emit another help_open and a help_repeat event.
    fireEvent.click(helpButton);

    const helpOpenCalls = mockSendRulesUxEvent.mock.calls.filter(
      ([arg]: any) => arg.type === 'rules_help_open'
    );
    const helpRepeatCalls = mockSendRulesUxEvent.mock.calls.filter(
      ([arg]: any) => arg.type === 'rules_help_repeat'
    );
    const weirdStateHelpCalls = mockSendRulesUxEvent.mock.calls.filter(
      ([arg]: any) => arg.type === 'rules_weird_state_help'
    );

    expect(helpOpenCalls.length).toBeGreaterThanOrEqual(2);
    expect(helpRepeatCalls.length).toBeGreaterThanOrEqual(1);
    expect(weirdStateHelpCalls.length).toBe(2);

    const repeatEvent = helpRepeatCalls[helpRepeatCalls.length - 1][0];
    expect(repeatEvent.repeatCount).toBeGreaterThanOrEqual(2);
    expect(repeatEvent.topic).toBe('forced_elimination');
    expect(repeatEvent.boardType).toBe('square8');
    expect(repeatEvent.numPlayers).toBe(2);

    // Spec-aligned telemetry: banner impression + details open with full weird-state context.
    const specEvents = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
    const impression = specEvents.find((event) => event.type === 'weird_state_banner_impression');
    const details = specEvents.find((event) => event.type === 'weird_state_details_open');

    expect(impression).toBeDefined();
    expect(details).toBeDefined();

    expect(details).toMatchObject({
      type: 'weird_state_details_open',
      source: 'hud',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'forced_elimination',
      reasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
      rulesContext: 'anm_forced_elimination',
      weirdStateType: 'forced-elimination',
    });

    expect(typeof details!.overlaySessionId).toBe('string');
    // Banner impression and details open should share the same overlaySessionId.
    expect(details!.overlaySessionId).toBe(impression!.overlaySessionId);
  });

  it('emits rules_help_open when territory help is opened during territory_processing', () => {
    const hud: HUDViewModel = {
      phase: {
        phaseKey: 'territory_processing' as any,
        label: 'Territory Processing',
        description: 'Resolve disconnected regions and territory.',
        icon: '▣',
        colorClass: 'bg-emerald-500',
        actionHint: 'Choose a region to process',
        spectatorHint: 'Player is resolving territory',
      },
      players: [],
      turnNumber: 5,
      moveNumber: 10,
      connectionStatus: 'connected' as any,
      isConnectionStale: false,
      isSpectator: false,
      spectatorCount: 0,
      subPhaseDetail: undefined,
      weirdState: undefined,
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
      pieRuleSummary: undefined,
      instruction: 'Instruction',
    };

    render(
      <GameHUD
        viewModel={hud}
        timeControl={undefined}
        rulesUxContext={{
          boardType: 'square8' as any,
          numPlayers: 2,
          aiDifficulty: 5,
        }}
      />
    );

    const helpButton = screen.getByTestId('hud-territory-help');
    fireEvent.click(helpButton);

    expect(
      mockSendRulesUxEvent.mock.calls.some(
        ([arg]: any) => arg.type === 'rules_help_open' && arg.topic === 'territory'
      )
    ).toBe(true);
  });
});
