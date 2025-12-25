import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { VictoryModal, type RematchStatus } from '../components/VictoryModal';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import { RingPlacementCountDialog } from '../components/RingPlacementCountDialog';
import { Button } from '../components/ui/Button';
import { StatusBanner } from '../components/ui/StatusBanner';
import { BackendBoardSection } from '../components/backend/BackendBoardSection';
import { BackendGameSidebar } from '../components/backend/BackendGameSidebar';
import {
  ScreenReaderAnnouncer,
  useGameAnnouncements,
  useGameStateAnnouncements,
} from '../components/ScreenReaderAnnouncer';
import { GameState, GameResult, Position, positionToString } from '../../shared/types/game';
import {
  toBoardViewModel,
  toEventLogViewModel,
  toHUDViewModel,
  toVictoryViewModel,
  deriveBoardDecisionHighlights,
} from '../adapters/gameViewModels';
import { useAuth } from '../contexts/AuthContext';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { useGame } from '../contexts/GameContext';
import { getGameOverBannerText } from '../utils/gameCopy';
import {
  buildGameEndExplanationFromEngineView,
  type GameEndEngineView,
  type GameEndExplanation,
  type GameEndPlayerScoreBreakdown,
  type GameEndRulesContextTag,
  type GameEndWeirdStateContext,
} from '../../shared/engine/gameEndExplanation';
import { getWeirdStateReasonForGameResult } from '../../shared/engine/weirdStateReasons';
import { useGameState } from '../hooks/useGameState';
import { useGameConnection } from '../hooks/useGameConnection';
import { useGameActions, useChatMessages, type PartialMove } from '../hooks/useGameActions';
import { useDecisionCountdown } from '../hooks/useDecisionCountdown';
import { useAutoMoveAnimation } from '../hooks/useMoveAnimation';
import { useInvalidMoveFeedback } from '../hooks/useInvalidMoveFeedback';
import { useIsMobile } from '../hooks/useIsMobile';
import { useGameSoundEffects } from '../hooks/useGameSoundEffects';
import { useGlobalGameShortcuts } from '../hooks/useKeyboardNavigation';
import { useSoundOptional } from '../contexts/SoundContext';
import type { PlayerChoice } from '../../shared/types/game';
import type { DecisionAutoResolvedMeta } from '../../shared/types/websocket';

// Import extracted hooks
import { useBackendBoardSelection } from '../hooks/useBackendBoardSelection';
import { useBackendBoardHandlers } from '../hooks/useBackendBoardHandlers';
import { useBackendGameStatus } from '../hooks/useBackendGameStatus';
import { useBackendChat } from '../hooks/useBackendChat';
import { useBackendTelemetry } from '../hooks/useBackendTelemetry';
import { useBackendConnectionShell } from '../hooks/useBackendConnectionShell';
import { useBackendDiagnosticsLog } from '../hooks/useBackendDiagnosticsLog';
import { useBackendDecisionUI } from '../hooks/useBackendDecisionUI';

/**
 * Get friendly display name for AI difficulty level with description.
 * Kept aligned with the ladder used elsewhere in the client.
 */
function getAIDifficultyLabel(difficulty: number): { label: string; color: string } {
  if (difficulty <= 2) return { label: 'Beginner', color: 'text-green-400' };
  if (difficulty <= 5) return { label: 'Intermediate', color: 'text-blue-400' };
  if (difficulty <= 8) return { label: 'Advanced', color: 'text-purple-400' };
  return { label: 'Expert', color: 'text-red-400' };
}

function renderGameHeader(gameState: GameState) {
  const playerSummary = gameState.players
    .sort((a, b) => a.playerNumber - b.playerNumber)
    .map((p) => {
      if (p.type === 'ai') {
        const difficulty = p.aiProfile?.difficulty ?? p.aiDifficulty ?? 5;
        const diffLabel = getAIDifficultyLabel(difficulty);
        return `${p.username || `AI-${p.playerNumber}`} (AI ${diffLabel.label} Lv${difficulty})`;
      }
      return `${p.username || `P${p.playerNumber}`} (Human)`;
    })
    .join(', ');

  return (
    <>
      <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold flex items-center gap-2">
        <img src="/ringrift-icon.png" alt="RingRift" className="w-6 h-6 sm:w-8 sm:h-8" />
        Game
      </h1>
      <p className="text-xs sm:text-sm text-gray-500 hidden sm:block">
        Board: {gameState.boardType} • {playerSummary}
      </p>
      <p className="text-xs text-gray-500 sm:hidden">
        {gameState.boardType} • {gameState.players.length}P
      </p>
    </>
  );
}

export interface BackendGameHostProps {
  /** Game id from the route (e.g. /game/:gameId or /spectate/:gameId) */
  gameId: string;
}
/**
 * BackendGameHost
 *
 * Host component for server-backed games (play + spectate). Owns:
 * - WebSocket lifecycle via GameContext
 * - Board interaction wiring for backend-valid moves
 * - HUD, event log, chat, and victory modal
 *
 * Rules semantics remain in the shared TS engine + orchestrator; this host
 * only orchestrates UI and backend integration.
 */
export const BackendGameHost: React.FC<BackendGameHostProps> = ({ gameId: routeGameId }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const {
    pendingRematchRequest,
    requestRematch,
    acceptRematch,
    declineRematch,
    rematchGameId,
    rematchLastStatus,
  } = useGame();

  // ConnectionShell: own connection lifecycle & status for the backend game
  const connection = useBackendConnectionShell(routeGameId);

  // Opponent disconnection signals (from useGameConnection, separate from shell)
  const { disconnectedOpponents, gameEndedByAbandonment } = useGameConnection();

  // GameStateController: subscribe to backend GameState and actions
  const {
    gameId,
    gameState,
    validMoves,
    victoryState,
    decisionAutoResolved,
    decisionPhaseTimeoutWarning,
    evaluationHistory,
  } = useGameState();
  const { submitMove } = useGameActions();
  const { messages: backendChatMessages, sendMessage: sendChatMessage } = useChatMessages();

  // Move animations - auto-detects moves from game state changes
  const { colorVisionMode, effectiveReducedMotion } = useAccessibility();
  const { pendingAnimation, clearAnimation } = useAutoMoveAnimation(gameState, {
    enabled: !effectiveReducedMotion,
  });

  // Invalid move feedback - shake animation and explanatory toasts
  const { shakingCellKey, triggerInvalidMove } = useInvalidMoveFeedback();

  const isMobile = useIsMobile();

  // DecisionUI: pending choice state + countdown timer
  const {
    pendingChoice,
    choiceDeadline,
    choiceTimeRemainingMs,
    respondToChoice,
    pendingChoiceView,
  } = useBackendDecisionUI();

  // Decision countdown: reconcile client-side timer with server timeout warnings.
  // The hook owns all reconciliation semantics between the client-local
  // baseline (from usePendingChoice) and any authoritative server warning.
  const decisionCountdown = useDecisionCountdown({
    pendingChoice,
    baseTimeRemainingMs: choiceTimeRemainingMs,
    timeoutWarning: decisionPhaseTimeoutWarning,
  });

  // Prefer the reconciled effective time when available, but retain a
  // conservative fallback to the local baseline for the ChoiceDialog so
  // that existing behaviour is preserved if the hook returns null.
  const reconciledDecisionTimeRemainingMs =
    decisionCountdown.effectiveTimeRemainingMs ?? choiceTimeRemainingMs ?? null;

  // DiagnosticsPanel: phase/player/choice + connection status logs
  const { eventLog, showSystemEventsInLog, setShowSystemEventsInLog } = useBackendDiagnosticsLog(
    gameState,
    pendingChoice,
    connection.connectionStatus,
    decisionAutoResolved,
    decisionPhaseTimeoutWarning,
    victoryState
  );

  const connectionStatus = connection.connectionStatus;
  const { isConnecting, error, lastHeartbeatAt } = connection;

  // ================== Extracted Hook: Board Selection ==================
  const boardSelection = useBackendBoardSelection(gameState, validMoves);
  const {
    selected,
    validTargets,
    mustMoveFrom: backendMustMoveFrom,
    chainCapturePath,
    setSelected,
    setValidTargets,
  } = boardSelection;

  // ================== Extracted Hook: Telemetry ==================
  const telemetry = useBackendTelemetry(gameState, victoryState, routeGameId);

  // ================== Extracted Hook: Game Status ==================
  const gameStatus = useBackendGameStatus({
    gameId,
    gameState,
    victoryState,
    routeGameId,
    weirdStateType: telemetry.weirdStateType,
    weirdStateFirstSeenAt: telemetry.weirdStateFirstSeenAt,
    weirdStateResignReported: telemetry.weirdStateResignReported,
    markWeirdStateResignReported: telemetry.markWeirdStateResignReported,
  });
  const {
    fatalGameError,
    setFatalGameError,
    isVictoryModalDismissed,
    dismissVictoryModal,
    isResigning,
    isResignConfirmOpen,
    setIsResignConfirmOpen,
    handleResign,
  } = gameStatus;

  // ================== Extracted Hook: Chat ==================
  const chat = useBackendChat(backendChatMessages, sendChatMessage);
  const { chatInput, setChatInput, messages: chatMessages, handleSubmit: handleChatSubmit } = chat;

  // Help / controls overlay
  const [showBoardControls, setShowBoardControls] = useState(false);

  // Sidebar density: keep advanced diagnostics available without overwhelming
  // new players. Persist preference locally per browser.
  const backendAdvancedSidebarStorageKey = 'ringrift_backend_sidebar_show_advanced';
  const [showAdvancedSidebarPanels, setShowAdvancedSidebarPanels] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      return localStorage.getItem(backendAdvancedSidebarStorageKey) === 'true';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(
        backendAdvancedSidebarStorageKey,
        showAdvancedSidebarPanels ? 'true' : 'false'
      );
    } catch {
      // Storage might be disabled; ignore.
    }
  }, [backendAdvancedSidebarStorageKey, showAdvancedSidebarPanels]);

  // Screen reader announcements for accessibility - using priority queue
  const { queue: announcementQueue, announce, removeAnnouncement } = useGameAnnouncements();

  // Derive current player info for announcements
  const currentPlayerForAnnouncements = gameState?.players.find(
    (p) => p.playerNumber === gameState.currentPlayer
  );
  const currentPlayerName =
    currentPlayerForAnnouncements?.username || `Player ${gameState?.currentPlayer ?? 1}`;
  const isLocalPlayerTurn = currentPlayerForAnnouncements?.id === user?.id;

  // Map victory reason to the type expected by GameAnnouncements
  const mapVictoryCondition = (
    reason: string | undefined
  ): 'elimination' | 'territory' | 'last_player_standing' => {
    switch (reason) {
      case 'ring_elimination':
        return 'elimination';
      case 'territory_control':
        return 'territory';
      default:
        return 'last_player_standing';
    }
  };

  // Use the automatic game state announcements hook
  useGameStateAnnouncements({
    currentPlayerName,
    isYourTurn: isLocalPlayerTurn,
    phase: gameState?.currentPhase,
    previousPhase: undefined, // Let the hook track this internally
    phaseDescription: undefined,
    timeRemaining: reconciledDecisionTimeRemainingMs,
    isGameOver: !!victoryState,
    winnerName:
      victoryState?.winner !== undefined
        ? gameState?.players.find((p) => p.playerNumber === victoryState.winner)?.username ||
          `Player ${victoryState.winner}`
        : undefined,
    victoryCondition: victoryState ? mapVictoryCondition(victoryState.reason) : undefined,
    isWinner:
      victoryState?.winner !== undefined &&
      gameState?.players.find((p) => p.playerNumber === victoryState.winner)?.id === user?.id,
    announce,
  });

  // Derived HUD state
  const currentPlayer = gameState?.players.find((p) => p.playerNumber === gameState.currentPlayer);
  const myPlayer = gameState?.players.find((p) => p.id === user?.id);
  const isPlayer = !!myPlayer;
  const isMyTurn = currentPlayer?.id === user?.id;

  // Sound effects for game events (phase changes, turns, moves, game end)
  useGameSoundEffects({
    gameState,
    victoryState,
    currentUserId: user?.id,
    myPlayerNumber: myPlayer?.playerNumber,
  });
  const sound = useSoundOptional();
  const isConnectionActive = connectionStatus === 'connected';

  useGlobalGameShortcuts({
    onShowHelp: () => {
      setShowBoardControls((prev) => !prev);
    },
    onResign: () => {
      if (!isPlayer || !gameState || gameState.gameStatus !== 'active') {
        return;
      }
      setIsResignConfirmOpen(true);
    },
    onToggleMute: () => {
      if (!sound) {
        return;
      }
      const nextMuted = !sound.muted;
      sound.toggleMute();
      toast(nextMuted ? 'Muted' : 'Sound on', { id: 'mute-toggle' });
    },
    onToggleFullscreen: () => {
      if (typeof document === 'undefined') return;

      if (document.fullscreenElement) {
        void document.exitFullscreen?.();
        return;
      }

      void document.documentElement.requestFullscreen?.();
    },
  });
  const boardInteractionMessage = (() => {
    if (!isPlayer) {
      return 'Moves disabled while spectating.';
    }
    if (!isConnectionActive) {
      return connectionStatus === 'reconnecting' || connectionStatus === 'connecting'
        ? 'Reconnecting to server…'
        : 'Disconnected from server.';
    }
    if (victoryState) {
      return 'Game completed.';
    }
    return null;
  })();

  const getInstruction = () => {
    if (!gameState || !currentPlayer) return undefined;
    if (!isPlayer) {
      return `Spectating: ${currentPlayer.username || `Player ${currentPlayer.playerNumber}`}'s turn`;
    }
    if (!isMyTurn) {
      return `Waiting for ${currentPlayer.username || `Player ${currentPlayer.playerNumber}`}...`;
    }

    // Surface an explicit warning when the server has indicated that the
    // current player's decision is approaching auto-resolution.
    if (
      decisionPhaseTimeoutWarning &&
      decisionPhaseTimeoutWarning.data.playerNumber === currentPlayer.playerNumber
    ) {
      const seconds = Math.max(1, Math.round(decisionPhaseTimeoutWarning.data.remainingMs / 1000));
      return `This decision will be auto-resolved in about ${seconds} second${
        seconds === 1 ? '' : 's'
      } if you do not respond.`;
    }

    switch (gameState.currentPhase) {
      case 'ring_placement':
        return 'Place rings on an empty cell or on top of an existing stack.';
      case 'movement':
        return 'Select a stack to move.';
      case 'capture':
        return 'Select a stack to capture with.';
      case 'chain_capture':
        return 'Chain capture in progress – select next capture target.';
      case 'line_processing':
        return 'Line processing – choose how to resolve your completed line.';
      case 'territory_processing':
        return 'Territory processing – resolve disconnected regions.';
      case 'forced_elimination':
        return 'No legal moves available – select a stack to eliminate from.';
      default:
        return 'Make your move.';
    }
  };

  // Placeholder hook for future game_error events (kept for parity with previous GamePage logic)
  useEffect(() => {
    if (!routeGameId) return;

    const handleGameError = (data: { data?: { message?: string; technical?: string } }) => {
      if (data && data.data) {
        setFatalGameError({
          message: data.data.message || 'An error occurred during the game.',
          technical: data.data.technical,
        });

        if (process.env.NODE_ENV === 'development' && data.data.technical) {
          console.error('[Game Error]', data.data.technical);
        }
      }
    };

    void handleGameError;

    return () => {
      // no-op cleanup; placeholder for future wiring
    };
  }, [routeGameId, setFatalGameError]);

  // When a rematch has been accepted and the server provides a new gameId,
  // navigate to the fresh backend game route. The GameProvider connection
  // shell will tear down the old WebSocket and establish a new one based
  // on the updated :gameId route param.
  useEffect(() => {
    if (rematchGameId) {
      navigate(`/game/${rematchGameId}`);
    }
  }, [rematchGameId, navigate]);

  // Diagnostics for phase / choice / connection are handled by useBackendDiagnosticsLog.
  // Choice countdown is owned by useBackendDecisionUI.
  //
  // We handle most keyboard shortcuts via useGlobalGameShortcuts. Keep a
  // narrow Escape handler here so the overlay closes even if it is mocked
  // in tests or rendered outside the focus trap.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;

      const target = event.target as HTMLElement | null;
      if (target) {
        const tagName = target.tagName;
        const isEditableTag = tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT';
        const isContentEditable = (target as HTMLElement).isContentEditable;
        if (isEditableTag || isContentEditable) {
          return;
        }
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
  }, [showBoardControls]);

  // ================== Extracted Hook: Board Handlers ==================
  const boardHandlers = useBackendBoardHandlers({
    gameState,
    validMoves,
    selected,
    validTargets,
    mustMoveFrom: backendMustMoveFrom,
    setSelected,
    setValidTargets,
    submitMove,
    isPlayer,
    isConnectionActive,
    isMyTurn,
    triggerInvalidMove,
  });
  const {
    ringPlacementCountPrompt,
    handleCellClick: handleBackendCellClick,
    handleCellDoubleClick: handleBackendCellDoubleClick,
    handleCellContextMenu: handleBackendCellContextMenu,
    handleConfirmRingPlacementCount,
    closeRingPlacementPrompt,
  } = boardHandlers;

  // Early-loading states
  if (isConnecting && !gameState) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-2">Connecting to game…</h1>
        <p className="text-sm text-gray-500">Game ID: {routeGameId}</p>
      </div>
    );
  }

  const reconnectionBanner =
    connectionStatus !== 'connected' && gameState ? (
      <StatusBanner
        variant="warning"
        icon={
          <div className="animate-spin h-4 w-4 border-2 border-amber-500 border-t-transparent rounded-full" />
        }
        actions={
          <>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={connection.reconnect}
              disabled={isConnecting}
            >
              Retry
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={() => navigate('/lobby')}>
              Lobby
            </Button>
          </>
        }
        className="mb-4"
      >
        {connectionStatus === 'reconnecting'
          ? 'Connection lost. Attempting to reconnect…'
          : connectionStatus === 'connecting'
            ? 'Connecting to game server…'
            : 'Disconnected from server. Moves are paused.'}
      </StatusBanner>
    ) : null;

  if (error && !gameState) {
    return (
      <div className="container mx-auto px-4 py-8 space-y-3">
        <h1 className="text-2xl font-bold mb-2">Unable to load game</h1>
        <p className="text-sm text-red-400">{error}</p>
        <p className="text-xs text-gray-500">Game ID: {routeGameId}</p>
      </div>
    );
  }

  if (!gameState || !gameId) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-2">Game not available</h1>
        <p className="text-sm text-gray-500">No game state received from server.</p>
      </div>
    );
  }

  const board = gameState.board;
  const boardType = gameState.boardType;

  const rulesUxContext = {
    boardType,
    numPlayers: gameState.players.length,
    aiDifficulty: (() => {
      const aiPlayers = gameState.players.filter((p) => p.type === 'ai');
      let maxDifficulty = 0;
      for (const p of aiPlayers) {
        const d = p.aiProfile?.difficulty ?? p.aiDifficulty;
        if (typeof d === 'number' && Number.isFinite(d) && d > maxDifficulty) {
          maxDifficulty = d;
        }
      }
      return maxDifficulty > 0 ? maxDifficulty : undefined;
    })(),
  } as const;

  // Derive decision-phase board highlights (if any) from the current GameState
  // and pending PlayerChoice, then feed them into the BoardViewModel so the
  // BoardView can render consistent geometry overlays for all roles.
  const baseDecisionHighlights = deriveBoardDecisionHighlights(gameState, pendingChoice);

  // Optional-capture visibility (backend): when the game is in capture phase
  // with no explicit PlayerChoice but canonical overtaking_capture moves
  // exist for the current player, synthesise a lightweight capture_direction
  // highlight model so both the over-taken stack and landing cells pulse
  // clearly, mirroring the sandbox host semantics.
  let decisionHighlights = baseDecisionHighlights;
  if (
    !decisionHighlights &&
    gameState.gameStatus === 'active' &&
    gameState.currentPhase === 'capture' &&
    Array.isArray(validMoves) &&
    validMoves.length > 0
  ) {
    const captureMoves = validMoves.filter((m) => m.type === 'overtaking_capture');
    if (captureMoves.length > 0) {
      type CaptureHighlight = { positionKey: string; intensity: 'primary' | 'secondary' };
      const highlights: CaptureHighlight[] = [];
      const seenPrimary = new Set<string>();
      const seenAny = new Set<string>();

      for (const move of captureMoves) {
        const target = move.captureTarget as Position | undefined;
        const landing = move.to as Position | undefined;

        if (landing) {
          const key = positionToString(landing);
          if (!seenPrimary.has(key)) {
            seenPrimary.add(key);
            seenAny.add(key);
            highlights.push({ positionKey: key, intensity: 'primary' });
          }
        }

        if (target) {
          const key = positionToString(target);
          if (!seenAny.has(key)) {
            seenAny.add(key);
            highlights.push({ positionKey: key, intensity: 'secondary' });
          }
        }
      }

      if (highlights.length > 0) {
        decisionHighlights = {
          choiceKind: 'capture_direction',
          highlights,
        };
      }
    }
  }

  const backendBoardViewModel = toBoardViewModel(board, {
    selectedPosition: selected || backendMustMoveFrom,
    validTargets,
    decisionHighlights,
    colorVisionMode,
  });

  let hudViewModel = toHUDViewModel(gameState, {
    instruction: getInstruction(),
    connectionStatus,
    lastHeartbeatAt,
    isSpectator: !isPlayer,
    currentUserId: user?.id,
    colorVisionMode,
    pendingChoice,
    choiceDeadline,
    choiceTimeRemainingMs: reconciledDecisionTimeRemainingMs,
    decisionIsServerCapped: decisionCountdown.isServerCapped,
    victoryState,
    // Pass LPS tracking from server GameState for UI display (RR-CANON-R172)
    lpsTracking: gameState.lpsTracking,
  });

  // Optional-capture HUD chip (backend): when capture is available directly
  // from the capture phase (with skip_capture as an option) but no explicit
  // decision choice is active, surface a bright attention chip so players do
  // not miss the opportunity. This mirrors the sandbox host behaviour.
  if (
    hudViewModel &&
    !hudViewModel.decisionPhase &&
    gameState.gameStatus === 'active' &&
    gameState.currentPhase === 'capture' &&
    Array.isArray(validMoves) &&
    validMoves.length > 0
  ) {
    const hasCaptureMove = validMoves.some((m) => m.type === 'overtaking_capture');
    const hasSkipCaptureMove = validMoves.some((m) => m.type === 'skip_capture');

    if (hasCaptureMove) {
      const actingVm = hudViewModel.players.find((p) => p.isCurrentPlayer);
      const actingPlayerNumber = actingVm?.playerNumber ?? gameState.currentPlayer;
      const actingPlayerName =
        actingVm?.username ||
        gameState.players.find((p) => p.playerNumber === actingPlayerNumber)?.username ||
        `Player ${actingPlayerNumber}`;

      hudViewModel = {
        ...hudViewModel,
        decisionPhase: {
          isActive: true,
          actingPlayerNumber,
          actingPlayerName,
          isLocalActor: !!(actingVm?.isUserPlayer && isMyTurn),
          label: hasSkipCaptureMove
            ? 'Optional capture available'
            : 'Capture available from this stack',
          description: hasSkipCaptureMove
            ? 'You may jump over a neighbouring stack for an overtaking capture, or skip capture to continue this turn.'
            : 'You may jump over a neighbouring stack for an overtaking capture from your last move.',
          shortLabel: 'Capture opportunity',
          timeRemainingMs: null,
          showCountdown: false,
          warningThresholdMs: undefined,
          isServerCapped: undefined,
          spectatorLabel: hasSkipCaptureMove
            ? `${actingPlayerName} may choose an overtaking capture or skip.`
            : `${actingPlayerName} may choose an overtaking capture.`,
          statusChip: {
            text: hasSkipCaptureMove
              ? 'Capture available – click a landing or skip'
              : 'Capture available – click a landing',
            tone: 'attention',
          },
          canSkip: hasSkipCaptureMove,
        },
      };
    }
  }

  const gameEndExplanation: GameEndExplanation | null = useMemo(() => {
    if (!victoryState || !gameState) {
      return null;
    }

    const mapOutcomeType = (reason: GameResult['reason']) => {
      switch (reason) {
        case 'ring_elimination':
          return 'ring_elimination' as const;
        case 'territory_control':
          return 'territory_control' as const;
        case 'last_player_standing':
          return 'last_player_standing' as const;
        case 'game_completed':
          return 'structural_stalemate' as const;
        case 'timeout':
          return 'timeout' as const;
        case 'resignation':
          return 'resignation' as const;
        case 'abandonment':
          return 'abandonment' as const;
        default:
          return null;
      }
    };

    const mapVictoryReasonCode = (reason: GameResult['reason']) => {
      switch (reason) {
        case 'ring_elimination':
          return 'victory_ring_majority' as const;
        case 'territory_control':
          return 'victory_territory_majority' as const;
        case 'last_player_standing':
          return 'victory_last_player_standing' as const;
        case 'game_completed':
          return 'victory_structural_stalemate_tiebreak' as const;
        case 'timeout':
          return 'victory_timeout' as const;
        case 'resignation':
          return 'victory_resignation' as const;
        case 'abandonment':
          return 'victory_abandonment' as const;
        default:
          return null;
      }
    };

    const outcomeType = mapOutcomeType(victoryState.reason);
    const victoryReasonCode = mapVictoryReasonCode(victoryState.reason);
    if (!outcomeType || !victoryReasonCode) {
      return null;
    }

    // Map numeric winner → player id (or null for draws/abandonment).
    const winnerPlayerId =
      victoryState.winner !== undefined
        ? (gameState.players.find((p) => p.playerNumber === victoryState.winner)?.id ?? null)
        : null;

    // Build per-player score breakdown from finalScore + board markers.
    const markersByPlayerNumber: Record<number, number> = {};
    for (const marker of gameState.board.markers.values()) {
      const key = marker.player;
      markersByPlayerNumber[key] = (markersByPlayerNumber[key] ?? 0) + 1;
    }

    const scoreBreakdown: Record<string, GameEndPlayerScoreBreakdown> = {};
    for (const player of gameState.players) {
      const num = player.playerNumber;
      const playerId = player.id;
      scoreBreakdown[playerId] = {
        playerId,
        eliminatedRings: victoryState.finalScore.ringsEliminated[num] ?? 0,
        territorySpaces: victoryState.finalScore.territorySpaces[num] ?? 0,
        markers: markersByPlayerNumber[num] ?? 0,
      };
    }

    // Derive weird-state context and rules-context tags using shared helpers.
    const weirdInfo = getWeirdStateReasonForGameResult(victoryState);
    let weirdStateContext: GameEndWeirdStateContext | undefined;
    let telemetryTags: GameEndRulesContextTag[] | undefined;

    if (weirdInfo) {
      weirdStateContext = {
        reasonCodes: [weirdInfo.reasonCode],
        primaryReasonCode: weirdInfo.reasonCode,
        rulesContextTags: [weirdInfo.rulesContext],
      };
      telemetryTags = [weirdInfo.rulesContext];
    }

    const engineView: GameEndEngineView = {
      gameId,
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      winnerPlayerId,
      outcomeType,
      victoryReasonCode,
      scoreBreakdown,
    };

    if (weirdStateContext) {
      engineView.weirdStateContext = weirdStateContext;
    }

    return buildGameEndExplanationFromEngineView(engineView, {
      teaching: undefined,
      telemetryTags,
      uxCopy: {
        shortSummaryKey: `game_end.${victoryState.reason}.short`,
        detailedSummaryKey: undefined,
      },
    });
  }, [victoryState, gameState, gameId]);

  const victoryViewModel = toVictoryViewModel(victoryState, gameState.players, gameState, {
    currentUserId: user?.id,
    isDismissed: isVictoryModalDismissed,
    colorVisionMode,
    gameEndExplanation,
  });

  const rematchStatus: RematchStatus | undefined = useMemo(() => {
    if (pendingRematchRequest) {
      const isRequester = user?.id === pendingRematchRequest.requesterId;
      return {
        isPending: true,
        requestId: pendingRematchRequest.id,
        requesterUsername: pendingRematchRequest.requesterUsername,
        isRequester,
        expiresAt: pendingRematchRequest.expiresAt,
        status: 'pending',
      };
    }

    if (rematchLastStatus) {
      return {
        isPending: false,
        status: rematchLastStatus,
      };
    }

    return undefined;
  }, [pendingRematchRequest, rematchLastStatus, user?.id]);

  const hudCurrentPlayer = hudViewModel.players.find((p) => p.isCurrentPlayer);

  const backendSelectedStackDetails = (() => {
    if (!backendBoardViewModel || !selected) return null;
    const key = positionToString(selected);
    const cell = backendBoardViewModel.cells.find((c) => c.positionKey === key);
    const stack = cell?.stack;
    if (!stack) return null;
    return {
      height: stack.stackHeight,
      cap: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  })();

  const gameOverBannerText =
    victoryState && isVictoryModalDismissed && victoryState.reason
      ? getGameOverBannerText(victoryState.reason)
      : null;

  return (
    <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-3 sm:space-y-4">
      {/* Screen reader live region for game announcements */}
      <ScreenReaderAnnouncer queue={announcementQueue} onAnnouncementSpoken={removeAnnouncement} />

      {reconnectionBanner}

      {/* Opponent disconnection banner - shown when opponents have disconnected but game continues */}
      {disconnectedOpponents.length > 0 && !gameEndedByAbandonment && !victoryState && (
        <StatusBanner
          variant="warning"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          }
          actions={<div className="animate-pulse h-2 w-2 rounded-full bg-amber-300" />}
          className="mb-4"
        >
          {disconnectedOpponents.length === 1
            ? `${disconnectedOpponents[0].username || 'A player'} has disconnected. Waiting for reconnection…`
            : `${disconnectedOpponents.length} players have disconnected. Waiting for reconnection…`}
        </StatusBanner>
      )}

      {/* Abandonment banner - shown when game ended due to reconnection timeout */}
      {gameEndedByAbandonment && (
        <StatusBanner
          variant="error"
          title="Game ended by abandonment"
          actions={
            <Button type="button" variant="secondary" size="sm" onClick={() => navigate('/lobby')}>
              Lobby
            </Button>
          }
          className="mb-4"
        >
          A player failed to reconnect within the allowed time.
        </StatusBanner>
      )}

      {gameOverBannerText && (
        <StatusBanner variant="success" className="mb-2">
          {gameOverBannerText}
        </StatusBanner>
      )}

      {fatalGameError && (
        <StatusBanner
          variant="error"
          title={fatalGameError.message}
          actions={
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => setFatalGameError(null)}
            >
              Dismiss
            </Button>
          }
        >
          {process.env.NODE_ENV === 'development' && fatalGameError.technical ? (
            <div className="text-xs font-mono text-red-200/90">
              Technical: {fatalGameError.technical}
            </div>
          ) : null}
        </StatusBanner>
      )}

      <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          {renderGameHeader(gameState)}
          {!isPlayer && (
            <span className="px-2 py-0.5 bg-purple-900/50 border border-purple-500/50 text-purple-200 text-xs rounded-full uppercase tracking-wider font-bold">
              Spectating
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
          <span className="hidden sm:inline">Status: {gameState.gameStatus}</span>
          <span className="hidden sm:inline">• Phase: {hudViewModel.phase.label}</span>
          <span className="hidden lg:inline">
            • Current:{' '}
            {hudCurrentPlayer
              ? hudCurrentPlayer.username || `P${hudCurrentPlayer.playerNumber}`
              : '—'}
          </span>
          {!isPlayer && (
            <button
              type="button"
              onClick={() => navigate('/lobby')}
              className="px-3 py-1 rounded bg-slate-800 border border-slate-600 text-[11px] text-slate-100 hover:bg-slate-700 hover:border-slate-400 transition-colors"
            >
              Back to lobby
            </button>
          )}
        </div>
      </header>

      <VictoryModal
        isOpen={!!victoryState && !isVictoryModalDismissed}
        viewModel={victoryViewModel}
        gameEndExplanation={gameEndExplanation}
        onClose={dismissVictoryModal}
        onReturnToLobby={() => navigate('/lobby')}
        onRequestRematch={() => {
          requestRematch();
        }}
        onAcceptRematch={(requestId) => {
          acceptRematch(requestId);
        }}
        onDeclineRematch={(requestId) => {
          declineRematch(requestId);
        }}
        rematchStatus={rematchStatus}
        currentUserId={user?.id}
      />

      <main className="flex flex-col lg:flex-row lg:gap-8 gap-4">
        {/* Board container - centers on mobile, takes available space on desktop */}
        <BackendBoardSection
          boardType={boardType}
          board={board}
          viewModel={backendBoardViewModel}
          selectedPosition={selected || backendMustMoveFrom}
          validTargets={validTargets}
          isSpectator={!isPlayer}
          pendingAnimation={pendingAnimation ?? undefined}
          chainCapturePath={chainCapturePath}
          shakingCellKey={shakingCellKey}
          onCellClick={(pos) => handleBackendCellClick(pos, board)}
          onCellDoubleClick={(pos) => handleBackendCellDoubleClick(pos, board)}
          onCellContextMenu={(pos) => handleBackendCellContextMenu(pos, board)}
          onAnimationComplete={clearAnimation}
          onShowBoardControls={() => setShowBoardControls(true)}
        />

        <BackendGameSidebar
          hudViewModel={hudViewModel}
          gameState={gameState}
          boardType={boardType}
          timeControl={gameState.timeControl}
          isMobile={isMobile}
          isPlayer={isPlayer}
          isMyTurn={isMyTurn}
          isConnectionActive={isConnectionActive}
          rulesUxContext={rulesUxContext}
          selectedPosition={selected}
          selectedStackDetails={backendSelectedStackDetails}
          boardInteractionMessage={boardInteractionMessage}
          pendingChoice={pendingChoice}
          pendingChoiceView={pendingChoiceView}
          choiceDeadline={choiceDeadline}
          reconciledDecisionTimeRemainingMs={reconciledDecisionTimeRemainingMs}
          isDecisionServerCapped={decisionCountdown.isServerCapped}
          decisionAutoResolved={decisionAutoResolved}
          moveHistory={gameState.moveHistory}
          currentMoveIndex={gameState.moveHistory.length - 1}
          eventLogViewModel={toEventLogViewModel(
            gameState.history,
            showSystemEventsInLog ? eventLog : [],
            victoryState,
            { maxEntries: 40 }
          )}
          showSystemEventsInLog={showSystemEventsInLog}
          isResigning={isResigning}
          isResignConfirmOpen={isResignConfirmOpen}
          showAdvancedSidebarPanels={showAdvancedSidebarPanels}
          gameId={gameId}
          hasVictoryState={!!victoryState}
          evaluationHistory={evaluationHistory}
          showSwapSidesPrompt={
            isPlayer &&
            isConnectionActive &&
            gameState.gameStatus === 'active' &&
            gameState.players.length === 2 &&
            gameState.rulesOptions?.swapRuleEnabled === true &&
            !!hudCurrentPlayer &&
            hudCurrentPlayer.playerNumber === gameState.currentPlayer &&
            hudCurrentPlayer.playerNumber === 2 &&
            !gameState.moveHistory.some((m) => m.type === 'swap_sides') &&
            gameState.moveHistory.some((m) => m.player === 1) &&
            !gameState.moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides')
          }
          hudCurrentPlayer={hudCurrentPlayer}
          chatMessages={chatMessages}
          chatInput={chatInput}
          onRespondToChoice={respondToChoice}
          onResign={handleResign}
          onResignConfirmOpenChange={setIsResignConfirmOpen}
          onSwapSides={() => {
            submitMove({
              type: 'swap_sides',
              to: { x: 0, y: 0 },
            } as PartialMove);
          }}
          onToggleSystemEventsInLog={() => setShowSystemEventsInLog((prev) => !prev)}
          onAdvancedPanelsToggle={(open) => setShowAdvancedSidebarPanels(open)}
          onHistoryError={(err) => {
            console.warn('Failed to load game history:', err.message);
          }}
          onChatInputChange={setChatInput}
          onChatSubmit={handleChatSubmit}
          onShowBoardControls={() => setShowBoardControls(true)}
        />
      </main>

      <RingPlacementCountDialog
        isOpen={!!ringPlacementCountPrompt}
        maxCount={ringPlacementCountPrompt?.maxCount ?? 1}
        defaultCount={
          ringPlacementCountPrompt?.hasStack
            ? 1
            : Math.min(2, ringPlacementCountPrompt?.maxCount ?? 1)
        }
        isStackPlacement={ringPlacementCountPrompt?.hasStack ?? false}
        onClose={closeRingPlacementPrompt}
        onConfirm={handleConfirmRingPlacementCount}
      />

      {showBoardControls && (
        <BoardControlsOverlay
          mode={isPlayer ? 'backend' : 'spectator'}
          onClose={() => setShowBoardControls(false)}
        />
      )}
    </div>
  );
};
