import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameEventLog } from '../components/GameEventLog';
import { SandboxTouchControlsPanel } from '../components/SandboxTouchControlsPanel';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import {
  BoardState,
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  positionToString,
  CreateGameRequest,
} from '../../shared/types/game';
import { useAuth } from '../contexts/AuthContext';
import { useSandbox, LocalConfig, LocalPlayerType } from '../contexts/SandboxContext';
import { useSandboxInteractions } from '../hooks/useSandboxInteractions';
import {
  toBoardViewModel,
  toEventLogViewModel,
  toVictoryViewModel,
  deriveBoardDecisionHighlights,
} from '../adapters/gameViewModels';
import { gameApi } from '../services/api';
import type { SandboxInteractionHandler } from '../sandbox/ClientSandboxEngine';
import { getGameOverBannerText } from '../utils/gameCopy';

const BOARD_PRESETS: Array<{
  value: BoardType;
  label: string;
  subtitle: string;
  blurb: string;
}> = [
  {
    value: 'square8',
    label: '8×8 Compact',
    subtitle: 'Fast tactical battles',
    blurb: 'Ideal for quick tests, fewer territories, emphasizes captures.',
  },
  {
    value: 'square19',
    label: '19×19 Classic',
    subtitle: 'Full RingRift experience',
    blurb: 'All line lengths and ring counts enabled for marathon sessions.',
  },
  {
    value: 'hexagonal',
    label: 'Full Hex',
    subtitle: 'High-mobility frontier',
    blurb: 'Hex adjacency, sweeping captures, and large territory swings.',
  },
];

const PLAYER_TYPE_META: Record<
  LocalPlayerType,
  { label: string; description: string; accent: string; chip: string }
> = {
  human: {
    label: 'Human',
    description: 'You control every move',
    accent: 'border-emerald-500 text-emerald-200',
    chip: 'bg-emerald-900/40 text-emerald-200',
  },
  ai: {
    label: 'Computer',
    description: 'Local heuristic AI',
    accent: 'border-sky-500 text-sky-200',
    chip: 'bg-sky-900/40 text-sky-200',
  },
};

const PHASE_COPY: Record<
  string,
  {
    label: string;
    summary: string;
  }
> = {
  ring_placement: {
    label: 'Ring Placement',
    summary: 'Place fresh stacks or build existing ones while keeping a legal move available.',
  },
  movement: {
    label: 'Movement',
    summary:
      'Pick a stack and travel a distance equal to its height, respecting board blocking rules.',
  },
  capture: {
    label: 'Capture',
    summary: 'Chain overtaking captures until no follow-up exists or a choice resolves.',
  },
  line_processing: {
    label: 'Line Processing',
    summary: 'Resolve completed lines and apply marker collapses/reward decisions.',
  },
  territory_processing: {
    label: 'Territory Processing',
    summary:
      'Evaluate disconnected regions, collapsing captured territory and enforcing self-elimination.',
  },
};

/**
 * Host component for the local sandbox experience.
 *
 * Responsibilities:
 * - Own sandbox configuration (board type, seats, player kinds) via SandboxContext
 * - Start sandbox games using ClientSandboxEngine, optionally attempting a backend game first
 * - Wire sandbox board interactions and local AI via useSandboxInteractions
 * - Render sandbox-specific HUD (players, selection, phase help, stall diagnostics)
 *
 * Rules semantics remain in the shared TS engine + orchestrator; this host only orchestrates
 * sandbox UI and engine lifecycle.
 */
export const SandboxGameHost: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const {
    config,
    setConfig,
    isConfigured,
    backendSandboxError,
    setBackendSandboxError,
    sandboxEngine,
    sandboxPendingChoice,
    setSandboxPendingChoice,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    sandboxCaptureTargets,
    setSandboxCaptureTargets,
    sandboxLastProgressAt,
    setSandboxLastProgressAt,
    sandboxStallWarning,
    setSandboxStallWarning,
    sandboxStateVersion,
    setSandboxStateVersion,
    initLocalSandboxEngine,
    resetSandboxEngine,
  } = useSandbox();

  // Local-only diagnostics / UX state
  const [isSandboxVictoryModalDismissed, setIsSandboxVictoryModalDismissed] = useState(false);

  // Selection + valid target highlighting
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);
  // Start with the movement grid overlay enabled by default; it helps
  // players understand valid moves and adjacency patterns.
  const [showMovementGrid, setShowMovementGrid] = useState(true);
  const [showValidTargetsOverlay, setShowValidTargetsOverlay] = useState(true);

  // Help / controls overlay for the active sandbox host
  const [showBoardControls, setShowBoardControls] = useState(false);

  const sandboxChoiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);

  const {
    handleCellClick: handleSandboxCellClick,
    handleCellDoubleClick: handleSandboxCellDoubleClick,
    handleCellContextMenu: handleSandboxCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection: clearSandboxSelection,
  } = useSandboxInteractions({
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    choiceResolverRef: sandboxChoiceResolverRef,
  });

  const handleSetupChange = (partial: Partial<LocalConfig>) => {
    setConfig((prev) => ({
      ...prev,
      ...partial,
      playerTypes: partial.numPlayers
        ? prev.playerTypes.map((t, idx) => (idx < partial.numPlayers! ? t : prev.playerTypes[idx]))
        : prev.playerTypes,
    }));
  };

  const handlePlayerTypeChange = (index: number, type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      next[index] = type;
      return { ...prev, playerTypes: next };
    });
  };

  const setAllPlayerTypes = (type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      for (let i = 0; i < prev.numPlayers; i += 1) {
        next[i] = type;
      }
      return { ...prev, playerTypes: next };
    });
  };

  const createSandboxInteractionHandler = (
    playerTypesSnapshot: LocalPlayerType[]
  ): SandboxInteractionHandler => {
    return {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const playerKind = playerTypesSnapshot[choice.playerNumber - 1] ?? 'human';

        // AI players: pick a random option without involving the UI.
        if (playerKind === 'ai') {
          const options = (choice as any).options as TChoice['options'];
          const optionsArray = (options as any[]) ?? [];
          if (optionsArray.length === 0) {
            throw new Error('SandboxInteractionHandler: no options available for AI choice');
          }
          const selectedOption = optionsArray[
            Math.floor(Math.random() * optionsArray.length)
          ] as TChoice['options'][number];

          return {
            choiceId: choice.id,
            playerNumber: choice.playerNumber,
            choiceType: choice.type,
            selectedOption,
          } as PlayerChoiceResponseFor<TChoice>;
        }

        // Human players
        if (choice.type === 'capture_direction') {
          const anyChoice = choice as any;
          const options = (anyChoice.options ?? []) as any[];
          const targets: Position[] = options.map((opt) => opt.landingPosition as Position);
          setSandboxCaptureChoice(choice);
          setSandboxCaptureTargets(targets);
        } else {
          setSandboxPendingChoice(choice);
        }

        return new Promise<PlayerChoiceResponseFor<TChoice>>((resolve) => {
          sandboxChoiceResolverRef.current = ((response: PlayerChoiceResponseFor<PlayerChoice>) => {
            resolve(response as PlayerChoiceResponseFor<TChoice>);
          }) as (response: PlayerChoiceResponseFor<PlayerChoice>) => void;
        });
      },
    };
  };

  const handleStartLocalGame = async () => {
    const nextBoardType = config.boardType;

    // First, attempt to create a real backend game using the same CreateGameRequest
    // shape as the lobby. On success, navigate into the real backend game route.
    try {
      const payload: CreateGameRequest = {
        boardType: nextBoardType,
        maxPlayers: config.numPlayers,
        isRated: false,
        isPrivate: true,
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        aiOpponents: (() => {
          const aiSeats = config.playerTypes
            .slice(0, config.numPlayers)
            .filter((t) => t === 'ai').length;
          if (aiSeats <= 0) return undefined;
          return {
            count: aiSeats,
            difficulty: Array(aiSeats).fill(5),
            mode: 'service',
            aiType: 'heuristic',
          };
        })(),
        // Mirror lobby behaviour: default-enable the pie rule for 2-player
        // backend sandbox games. Local-only sandbox games (fallback path)
        // continue to use the shared engine's defaults.
        rulesOptions:
          config.numPlayers === 2
            ? { swapRuleEnabled: true }
            : undefined,
      };

      const game = await gameApi.createGame(payload);
      navigate(`/game/${game.id}`);
      return;
    } catch (err) {
      console.error('Failed to create backend sandbox game, falling back to local-only board', err);
      setBackendSandboxError(
        'Backend sandbox game could not be created; falling back to local-only board only.'
      );
    }

    // Fallback: local sandbox engine using orchestrator-first semantics.
    const interactionHandler = createSandboxInteractionHandler(
      config.playerTypes.slice(0, config.numPlayers)
    );
    const engine = initLocalSandboxEngine({
      boardType: nextBoardType,
      numPlayers: config.numPlayers,
      playerTypes: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[],
      interactionHandler,
    });

    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);

    // If the first player is an AI, immediately start the sandbox AI turn loop.
    if (engine) {
      const state = engine.getGameState();
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (current && current.type === 'ai') {
        maybeRunSandboxAiIfNeeded();
      }
    }
  };

  const handleCopySandboxTrace = async () => {
    try {
      if (typeof window === 'undefined') {
        return;
      }

      const anyWindow = window as any;
      const trace = anyWindow.__RINGRIFT_SANDBOX_TRACE__ ?? [];
      const payload = JSON.stringify(trace, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox AI trace copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox AI trace', trace);
        toast.success('Sandbox AI trace logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox AI trace', err);
      toast.error('Failed to export sandbox AI trace; see console for details.');
    }
  };

  // Game view once configured (local sandbox)
  const sandboxGameState: GameState | null = sandboxEngine ? sandboxEngine.getGameState() : null;
  const sandboxBoardState: BoardState | null = sandboxGameState?.board ?? null;
  const sandboxVictoryResult = sandboxEngine ? sandboxEngine.getVictoryResult() : null;

  const sandboxGameOverBannerText =
    sandboxVictoryResult && isSandboxVictoryModalDismissed && sandboxVictoryResult.reason
      ? getGameOverBannerText(sandboxVictoryResult.reason)
      : null;

  const boardTypeValue = sandboxBoardState?.type ?? config.boardType;
  const boardPresetInfo = BOARD_PRESETS.find((preset) => preset.value === boardTypeValue);
  const boardDisplayLabel = boardPresetInfo?.label ?? boardTypeValue;
  const boardDisplaySubtitle = boardPresetInfo?.subtitle ?? 'Custom configuration';
  const boardDisplayBlurb =
    boardPresetInfo?.blurb ?? 'Custom layout selected for this local sandbox match.';

  const sandboxPlayersList =
    sandboxGameState?.players ??
    Array.from({ length: config.numPlayers }, (_, idx) => ({
      playerNumber: idx + 1,
      username: `Player ${idx + 1}`,
      type: config.playerTypes[idx] ?? 'human',
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));

  const sandboxCurrentPlayerNumber = sandboxGameState?.currentPlayer ?? 1;
  const sandboxCurrentPlayer =
    sandboxPlayersList.find((p) => p.playerNumber === sandboxCurrentPlayerNumber) ??
    sandboxPlayersList[0];

  const sandboxPhaseKey = sandboxGameState?.currentPhase ?? 'ring_placement';
  const sandboxPhaseDetails = PHASE_COPY[sandboxPhaseKey] ?? PHASE_COPY.ring_placement;

  const humanSeatCount = sandboxPlayersList.filter((p) => p.type === 'human').length;
  const aiSeatCount = sandboxPlayersList.length - humanSeatCount;

  // Derive board VM + HUD-like summaries
  const primaryValidTargets =
    sandboxCaptureTargets.length > 0 ? sandboxCaptureTargets : validTargets;

  const displayedValidTargets = showValidTargetsOverlay ? primaryValidTargets : [];

  // Derive decision-phase highlights from the current sandbox GameState and
  // whichever PlayerChoice is currently active. Capture-direction choices
  // take precedence over generic pending choices so that landing/target
  // geometry is always visible while the capture UI is open.
  const activePendingChoice: PlayerChoice | null = sandboxCaptureChoice ?? sandboxPendingChoice;

  const decisionHighlights =
    sandboxGameState && activePendingChoice
      ? deriveBoardDecisionHighlights(sandboxGameState, activePendingChoice)
      : undefined;

  const sandboxBoardViewModel = sandboxBoardState
    ? toBoardViewModel(sandboxBoardState, {
        selectedPosition: selected,
        validTargets: displayedValidTargets,
        decisionHighlights,
      })
    : null;

  const sandboxVictoryViewModel = sandboxVictoryResult
    ? toVictoryViewModel(
        sandboxVictoryResult,
        sandboxGameState?.players ?? [],
        sandboxGameState ?? undefined,
        {
          currentUserId: user?.id,
          isDismissed: isSandboxVictoryModalDismissed,
        }
      )
    : null;

  const sandboxHudPlayers = sandboxPlayersList.map((player) => ({
    playerNumber: player.playerNumber,
    username: player.username || `Player ${player.playerNumber}`,
    type: player.type,
    ringsInHand: player.ringsInHand,
    eliminatedRings: player.eliminatedRings,
    territorySpaces: player.territorySpaces,
    isCurrent: player.playerNumber === sandboxCurrentPlayerNumber,
  }));

  const sandboxModeNotes = [
    `Board: ${boardDisplayLabel}`,
    `${humanSeatCount} human seat${humanSeatCount === 1 ? '' : 's'} · ${aiSeatCount} AI`,
    sandboxEngine
      ? 'Engine parity mode with local AI and choice handler.'
      : 'Legacy local sandbox fallback (no backend).',
    'Runs entirely in-browser; use "Change Setup" to switch configurations.',
  ];

  const sandboxHudViewModel = {
    players: sandboxHudPlayers,
    phaseDetails: sandboxPhaseDetails,
    modeNotes: sandboxModeNotes,
  };

  const sandboxEventLogViewModel = toEventLogViewModel(
    sandboxGameState?.history ?? [],
    [],
    sandboxVictoryResult,
    { maxEntries: 40 }
  );

  const selectedStackDetails = (() => {
    if (!sandboxBoardState || !selected) return null;
    const key = positionToString(selected);
    const stack = sandboxBoardState.stacks.get(key);
    if (!stack) return null;
    return {
      height: stack.stackHeight,
      cap: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  })();

  const activePlayerTypes = config.playerTypes.slice(0, config.numPlayers);
  const setupHumanSeatCount = activePlayerTypes.filter((t) => t === 'human').length;
  const setupAiSeatCount = activePlayerTypes.length - setupHumanSeatCount;
  const selectedBoardPreset =
    BOARD_PRESETS.find((preset) => preset.value === config.boardType) ?? BOARD_PRESETS[0];

  // Keyboard shortcuts for sandbox overlay:
  // - "?" (Shift + "/") toggles the Board Controls overlay when a sandbox game is active.
  // - "Escape" closes the overlay when open.
  useEffect(() => {
    if (!isConfigured || !sandboxEngine) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;

      const target = event.target as HTMLElement | null;
      if (target) {
        const tagName = target.tagName;
        const isEditableTag =
          tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT';
        const isContentEditable = target.isContentEditable;
        if (isEditableTag || isContentEditable) {
          return;
        }
      }

      if (event.key === '?' || (event.key === '/' && event.shiftKey)) {
        event.preventDefault();
        setShowBoardControls((prev) => !prev);
        return;
      }

      if (event.key === 'Escape' && showBoardControls) {
        event.preventDefault();
        setShowBoardControls(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isConfigured, sandboxEngine, showBoardControls]);

  // Pre-game setup view
  if (!isConfigured || !sandboxEngine) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <div className="container mx-auto px-4 py-8 space-y-6">
          <header>
            <h1 className="text-3xl font-bold mb-1">Start a RingRift Game (Local Sandbox)</h1>
            <p className="text-sm text-slate-400">
              This mode runs entirely in the browser using a local board. To view or play a real
              server-backed game, navigate to a URL with a game ID (e.g.
              <code className="ml-1 text-xs text-slate-300">/game/:gameId</code>).
            </p>
          </header>

          <section className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <div className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 space-y-6 text-slate-100 shadow-lg">
            {backendSandboxError && (
              <div className="p-3 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded-lg">
                {backendSandboxError}
              </div>
            )}

            <div className="space-y-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-400">Players</p>
                  <h2 className="text-lg font-semibold text-white">Seats & control</h2>
                </div>
                <div className="flex gap-2 text-xs">
                  {[2, 3, 4].map((count) => (
                    <button
                      key={count}
                      type="button"
                      onClick={() => handleSetupChange({ numPlayers: count })}
                      className={`px-2 py-1 rounded-full border ${
                        config.numPlayers === count
                          ? 'border-emerald-400 text-emerald-200 bg-emerald-900/30'
                          : 'border-slate-600 text-slate-300 hover:border-slate-400'
                      }`}
                    >
                      {count} Players
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                {Array.from({ length: config.numPlayers }, (_, i) => {
                  const type = config.playerTypes[i];
                  const meta = PLAYER_TYPE_META[type];
                  return (
                    <div
                      key={i}
                      className={`rounded-xl border bg-slate-900/60 px-4 py-3 flex items-center justify-between gap-4 ${meta.accent}`}
                    >
                      <div>
                        <p className="text-sm font-semibold text-white">Player {i + 1}</p>
                        <p className="text-xs text-slate-300">{meta.description}</p>
                      </div>
                      <div className="flex gap-2">
                        {(['human', 'ai'] as LocalPlayerType[]).map((candidate) => {
                          const isActive = type === candidate;
                          return (
                            <button
                              key={candidate}
                              type="button"
                              onClick={() => handlePlayerTypeChange(i, candidate)}
                              className={`px-3 py-1 rounded-full border text-xs font-semibold transition ${
                                isActive
                                  ? 'border-white/80 text-white bg-white/10'
                                  : 'border-slate-600 text-slate-300 hover:border-slate-400'
                              }`}
                            >
                              {PLAYER_TYPE_META[candidate].label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex flex-wrap gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => setAllPlayerTypes('human')}
                  className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition"
                >
                  All Human
                </button>
                <button
                  type="button"
                  onClick={() => setAllPlayerTypes('ai')}
                  className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition"
                >
                  All AI
                </button>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-400">Board</p>
                  <h2 className="text-lg font-semibold text-white">Choose a layout</h2>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                {BOARD_PRESETS.map((preset) => {
                  const isSelected = preset.value === config.boardType;
                  return (
                    <button
                      key={preset.value}
                      type="button"
                      onClick={() => handleSetupChange({ boardType: preset.value })}
                      className={`p-4 text-left rounded-2xl border transition shadow-sm ${
                        isSelected
                          ? 'border-emerald-400 bg-emerald-900/20 text-white'
                          : 'border-slate-600 bg-slate-900/60 text-slate-200 hover:border-slate-400'
                      }`}
                    >
                      <span className="text-xs uppercase tracking-wide text-slate-400">
                        {preset.subtitle}
                      </span>
                      <p className="text-lg font-semibold">{preset.label}</p>
                      <p className="text-xs text-slate-300 mt-1">{preset.blurb}</p>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-700 text-slate-100 shadow-lg space-y-4">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-400">Summary</p>
              <h2 className="text-xl font-bold text-white">{selectedBoardPreset.label}</h2>
              <p className="text-sm text-slate-300">{selectedBoardPreset.blurb}</p>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Humans</span>
                <span className="font-semibold">{setupHumanSeatCount}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">AI opponents</span>
                <span className="font-semibold">{setupAiSeatCount}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Total seats</span>
                <span className="font-semibold">{config.numPlayers}</span>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-slate-400">
                We first attempt to stand up a backend game with these settings. If that fails, we
                fall back to a purely client-local sandbox so you can still test moves offline.
              </p>
              <button
                type="button"
                onClick={handleStartLocalGame}
                className="w-full px-4 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white shadow-lg shadow-emerald-900/40 transition"
              >
                Launch Game
              </button>
            </div>
          </div>
          </section>
        </div>
      </div>
    );
  }

  // === Active sandbox game ===
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-4 py-8 space-y-4">
        {sandboxStallWarning && (
        <div className="p-3 rounded-xl border border-amber-500/70 bg-amber-900/40 text-amber-100 text-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <span>{sandboxStallWarning}</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleCopySandboxTrace}
              className="px-3 py-1 rounded-lg border border-amber-300 bg-amber-800/70 text-[11px] font-semibold hover:border-amber-100 hover:bg-amber-700/80"
            >
              Copy AI trace
            </button>
            <button
              type="button"
              onClick={() => setSandboxStallWarning(null)}
              className="px-2 py-1 rounded-lg border border-slate-500 text-[11px] hover:border-slate-300"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

        {sandboxGameOverBannerText && (
        <div className="p-3 rounded-xl border border-emerald-500/70 bg-emerald-900/40 text-emerald-100 text-xs">
          {sandboxGameOverBannerText}
        </div>
      )}

        {sandboxGameState && (
        <VictoryModal
          isOpen={!!sandboxVictoryResult && !isSandboxVictoryModalDismissed}
          viewModel={sandboxVictoryViewModel}
          onClose={() => {
            setIsSandboxVictoryModalDismissed(true);
          }}
          onReturnToLobby={() => {
            resetSandboxEngine();
            setSelected(undefined);
            setValidTargets([]);
            setBackendSandboxError(null);
            setSandboxPendingChoice(null);
            setIsSandboxVictoryModalDismissed(false);
          }}
        />
      )}

        <ChoiceDialog
        choice={sandboxPendingChoice}
        deadline={null}
        onSelectOption={(choice, option) => {
          const resolver = sandboxChoiceResolverRef.current;
          if (resolver) {
            resolver({
              choiceId: choice.id,
              playerNumber: choice.playerNumber,
              choiceType: choice.type,
              selectedOption: option,
            } as PlayerChoiceResponseFor<PlayerChoice>);
            sandboxChoiceResolverRef.current = null;
          }
          setSandboxPendingChoice(null);
        }}
      />

        <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
        <section className="flex justify-center md:block">
          {sandboxBoardState && (
            <div className="inline-block space-y-3">
              <div className="p-4 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Local Sandbox</p>
                    <h1 className="text-2xl font-bold text-white">Game – {boardDisplayLabel}</h1>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        resetSandboxEngine();
                        setSelected(undefined);
                        setValidTargets([]);
                        setBackendSandboxError(null);
                        setSandboxPendingChoice(null);
                        setSandboxStallWarning(null);
                        setSandboxLastProgressAt(null);
                        setIsSandboxVictoryModalDismissed(false);
                      }}
                      className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                    >
                      Change Setup
                    </button>
                    <button
                      type="button"
                      aria-label="Show board controls"
                      data-testid="board-controls-button"
                      onClick={() => setShowBoardControls(true)}
                      className="h-8 w-8 rounded-full border border-slate-600 text-[11px] leading-none text-slate-200 hover:bg-slate-800/80"
                    >
                      ?
                    </button>
                  </div>
                </div>
              </div>

              {sandboxBoardState && sandboxBoardViewModel && (
                <BoardView
                  boardType={sandboxBoardState.type}
                  board={sandboxBoardState}
                  viewModel={sandboxBoardViewModel}
                  selectedPosition={selected}
                  validTargets={displayedValidTargets}
                  onCellClick={(pos) => handleSandboxCellClick(pos)}
                  onCellDoubleClick={(pos) => handleSandboxCellDoubleClick(pos)}
                  onCellContextMenu={(pos) => handleSandboxCellContextMenu(pos)}
                  showMovementGrid={showMovementGrid}
                  showCoordinateLabels={
                    sandboxBoardState.type === 'square8' || sandboxBoardState.type === 'square19'
                  }
                />
              )}

              <section className="mt-1 p-3 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 text-xs text-slate-200">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600">
                    {boardDisplaySubtitle}
                  </span>
                  <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600">
                    Players: {config.numPlayers} ({humanSeatCount} human, {aiSeatCount} AI)
                  </span>
                  <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600 min-w-[10rem] inline-flex justify-center text-center">
                    Phase: {sandboxPhaseDetails.label}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {sandboxPlayersList.map((player) => {
                    const typeKey = player.type === 'ai' ? 'ai' : 'human';
                    const meta = PLAYER_TYPE_META[typeKey as LocalPlayerType];
                    const isCurrent = player.playerNumber === sandboxCurrentPlayerNumber;
                    const nameLabel = player.username || `Player ${player.playerNumber}`;
                    return (
                      <span
                        key={player.playerNumber}
                        className={`px-3 py-1 rounded-full border transition ${
                          isCurrent ? 'border-white text-white bg-white/15' : meta.chip
                        }`}
                      >
                        P{player.playerNumber} • {nameLabel} ({meta.label})
                      </span>
                    );
                  })}
                </div>
              </section>
            </div>
          )}
        </section>

        <aside className="w-full md:w-80 space-y-4 text-sm text-slate-100">
          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
            <h2 className="font-semibold">Players</h2>
            <div className="space-y-2">
              {sandboxHudViewModel.players.map((player) => (
                <div
                  key={player.playerNumber}
                  className={`rounded-xl border px-3 py-2 text-xs flex items-center justify-between ${
                    player.isCurrent
                      ? 'border-emerald-400 bg-emerald-900/20'
                      : 'border-slate-700 bg-slate-900/40'
                  }`}
                >
                  <div>
                    <p className="font-semibold text-white">
                      P{player.playerNumber} {player.username ? `• ${player.username}` : ''}
                    </p>
                    <p className="text-[11px] text-slate-400">
                      {player.type === 'ai' ? 'Computer' : 'Human'}
                    </p>
                  </div>
                  <div className="flex gap-3 text-right">
                    <div>
                      <p className="text-sm font-bold text-white">{player.ringsInHand}</p>
                      <p className="text-[11px] text-slate-400">in hand</p>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-white">{player.territorySpaces}</p>
                      <p className="text-[11px] text-slate-400">territory</p>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-white">{player.eliminatedRings}</p>
                      <p className="text-[11px] text-slate-400">eliminated</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {sandboxEngine &&
            sandboxGameState &&
            sandboxGameState.gameStatus === 'active' &&
            sandboxGameState.players.length === 2 &&
            sandboxGameState.rulesOptions?.swapRuleEnabled === true &&
            sandboxEngine.canCurrentPlayerSwapSides() && (
              <div className="p-3 border border-amber-500/60 rounded-2xl bg-amber-900/40 text-xs space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-amber-100">
                    Pie rule available: swap colours with Player 1.
                  </span>
                  <button
                    type="button"
                    className="px-2 py-1 rounded bg-amber-500 hover:bg-amber-400 text-black font-semibold"
                    onClick={() => {
                      sandboxEngine.applySwapSidesForCurrentPlayer();
                      setSelected(undefined);
                      setValidTargets([]);
                      setSandboxPendingChoice(null);
                      setSandboxStateVersion((v) => v + 1);
                    }}
                  >
                    Swap colours
                  </button>
                </div>
                <p className="text-amber-100/80">
                  As Player 2, you may use this once, immediately after Player 1’s first turn.
                </p>
              </div>
            )}

          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60">
            <GameEventLog viewModel={sandboxEventLogViewModel} />
          </div>

          <SandboxTouchControlsPanel
            selectedPosition={selected}
            selectedStackDetails={selectedStackDetails}
            validTargets={primaryValidTargets}
            isCaptureDirectionPending={!!sandboxCaptureChoice}
            captureTargets={sandboxCaptureTargets}
            // Multi-segment capture undo is not yet exposed by the sandbox
            // engine; this remains a no-op until the underlying rules
            // pipeline supports segment-level rewind.
            canUndoSegment={false}
            onClearSelection={() => {
              clearSandboxSelection();
            }}
            onUndoSegment={() => {
              // no-op for now
            }}
            // For now, treat "Finish move" as an explicit selection reset that
            // clears highlights without issuing additional engine actions.
            onApplyMove={() => {
              clearSandboxSelection();
            }}
            showMovementGrid={showMovementGrid}
            onToggleMovementGrid={(next) => setShowMovementGrid(next)}
            showValidTargets={showValidTargetsOverlay}
            onToggleValidTargets={(next) => setShowValidTargetsOverlay(next)}
            phaseLabel={sandboxPhaseDetails.label}
          />

          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
            <h2 className="font-semibold">Phase Guide</h2>
            <p className="text-xs uppercase tracking-wide text-slate-400">
              {sandboxHudViewModel.phaseDetails.label}
            </p>
            <p className="text-sm text-slate-200">{sandboxHudViewModel.phaseDetails.summary}</p>
            <p className="text-xs text-slate-400">
              Complete the current requirement to advance the turn (chain captures, line rewards,
              etc.).
            </p>
          </div>

          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
            <h2 className="font-semibold">Sandbox Notes</h2>
            <ul className="list-disc list-inside text-slate-300 space-y-1 text-xs">
              {sandboxHudViewModel.modeNotes.map((note, idx) => (
                <li key={idx}>{note}</li>
              ))}
            </ul>
          </div>
        </aside>
        </main>

        {showBoardControls && (
        <BoardControlsOverlay
          mode="sandbox"
          hasTouchControlsPanel
          onClose={() => setShowBoardControls(false)}
        />
        )}
      </div>
    </div>
  );
};
