import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal, type RematchStatus } from '../components/VictoryModal';
import { GameHUD } from '../components/GameHUD';
import { MobileGameHUD } from '../components/MobileGameHUD';
import { GameEventLog } from '../components/GameEventLog';
import { MoveHistory } from '../components/MoveHistory';
import { GameHistoryPanel } from '../components/GameHistoryPanel';
import { EvaluationPanel } from '../components/EvaluationPanel';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import { ResignButton } from '../components/ResignButton';
import { RingPlacementCountDialog } from '../components/RingPlacementCountDialog';
import { Button } from '../components/ui/Button';
import { StatusBanner } from '../components/ui/StatusBanner';
import {
  ScreenReaderAnnouncer,
  useGameAnnouncements,
  useGameStateAnnouncements,
} from '../components/ScreenReaderAnnouncer';
import { gameApi } from '../services/api';
import {
  BoardState,
  GameState,
  GameResult,
  Move,
  Position,
  positionToString,
  positionsEqual,
} from '../../shared/types/game';
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
import { formatPosition } from '../../shared/engine/notation';
import {
  buildGameEndExplanationFromEngineView,
  type GameEndEngineView,
  type GameEndExplanation,
  type GameEndPlayerScoreBreakdown,
  type GameEndRulesContextTag,
  type GameEndWeirdStateContext,
} from '../../shared/engine/gameEndExplanation';
import {
  getWeirdStateReasonForGameResult,
  isSurfaceableWeirdStateType,
} from '../../shared/engine/weirdStateReasons';
import { useGameState } from '../hooks/useGameState';
import { getWeirdStateBanner } from '../utils/gameStateWeirdness';
import type { RulesUxWeirdStateType } from '../../shared/telemetry/rulesUxEvents';
import { sendRulesUxEvent } from '../utils/rulesUxTelemetry';
import {
  sendDifficultyCalibrationEvent,
  getDifficultyCalibrationSession,
  clearDifficultyCalibrationSession,
} from '../utils/difficultyCalibrationTelemetry';
import { useGameConnection } from '../hooks/useGameConnection';
import {
  useGameActions,
  usePendingChoice,
  useChatMessages,
  type PendingChoiceView,
  type PartialMove,
} from '../hooks/useGameActions';
import { useDecisionCountdown } from '../hooks/useDecisionCountdown';
import { useAutoMoveAnimation } from '../hooks/useMoveAnimation';
import {
  useInvalidMoveFeedback,
  analyzeInvalidMove as analyzeInvalid,
} from '../hooks/useInvalidMoveFeedback';
import { useIsMobile } from '../hooks/useIsMobile';
import { useGameSoundEffects } from '../hooks/useGameSoundEffects';
import { useGlobalGameShortcuts } from '../hooks/useKeyboardNavigation';
import { useSoundOptional } from '../contexts/SoundContext';
import type { PlayerChoice } from '../../shared/types/game';
import type {
  DecisionAutoResolvedMeta,
  DecisionPhaseTimeoutWarningPayload,
} from '../../shared/types/websocket';
import type { ConnectionStatus } from '../contexts/GameContext';
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

function describeDecisionAutoResolved(meta: DecisionAutoResolvedMeta): string {
  const playerLabel = `P${meta.actingPlayerNumber}`;
  const reasonLabel = meta.reason === 'timeout' ? 'timeout' : meta.reason.replace(/_/g, ' ');

  const choiceKindLabel = (() => {
    switch (meta.choiceKind) {
      case 'line_order':
        return 'line order';
      case 'line_reward':
        return 'line reward';
      case 'ring_elimination':
        return 'ring elimination';
      case 'territory_region_order':
        return 'territory region order';
      case 'capture_direction':
        return 'capture direction';
      default:
        return meta.choiceKind.replace(/_/g, ' ');
    }
  })();

  const movePart = meta.resolvedMoveId ? ` (moveId: ${meta.resolvedMoveId})` : '';

  return `Decision auto-resolved for ${playerLabel}: ${choiceKindLabel} (reason: ${reasonLabel})${movePart}`;
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

/**
 * Backend ConnectionShell: wraps useGameConnection and owns connect/disconnect
 * lifecycle for a specific :gameId route.
 */
interface BackendConnectionShellState {
  routeGameId: string;
  gameId: string | null;
  connectionStatus: ConnectionStatus;
  isConnecting: boolean;
  error: string | null;
  lastHeartbeatAt: number | null;
  reconnect: () => void;
}

function useBackendConnectionShell(routeGameId: string): BackendConnectionShellState {
  const { gameId, status, isConnecting, error, lastHeartbeatAt, connectToGame, disconnect } =
    useGameConnection();

  useEffect(() => {
    if (!routeGameId) {
      disconnect();
      return;
    }

    void connectToGame(routeGameId);

    return () => {
      disconnect();
    };
  }, [routeGameId, connectToGame, disconnect]);

  return {
    routeGameId,
    gameId,
    connectionStatus: status,
    isConnecting,
    error,
    lastHeartbeatAt,
    reconnect: () => {
      if (!routeGameId) {
        return;
      }
      void connectToGame(routeGameId);
    },
  };
}

/**
 * Backend DiagnosticsPanel hook: produces a rolling log of phase/player/choice
 * and connection-status events suitable for GameEventLog.
 */
interface BackendDiagnosticsState {
  eventLog: string[];
  showSystemEventsInLog: boolean;
  setShowSystemEventsInLog: React.Dispatch<React.SetStateAction<boolean>>;
}

function useBackendDiagnosticsLog(
  gameState: GameState | null,
  pendingChoice: PlayerChoice | null,
  connectionStatus: ConnectionStatus,
  decisionAutoResolved: DecisionAutoResolvedMeta | null,
  decisionPhaseTimeoutWarning: DecisionPhaseTimeoutWarningPayload | null,
  victoryState: GameResult | null
): BackendDiagnosticsState {
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [showSystemEventsInLog, setShowSystemEventsInLog] = useState(true);

  const lastPhaseRef = useRef<string | null>(null);
  const lastCurrentPlayerRef = useRef<number | null>(null);
  const lastChoiceIdRef = useRef<string | null>(null);
  const lastAutoResolvedKeyRef = useRef<string | null>(null);
  const lastConnectionStatusRef = useRef<ConnectionStatus | null>(null);
  const lastTimeoutWarningKeyRef = useRef<string | null>(null);
  const lastWeirdStateTypeRef = useRef<string | null>(null);
  const forcedElimContextRef = useRef<{
    active: boolean;
    startTotal: number;
    playerNumber: number | null;
  } | null>(null);

  // Phase / current player / choice transitions
  useEffect(() => {
    if (!gameState) {
      lastPhaseRef.current = null;
      lastCurrentPlayerRef.current = null;
      return;
    }

    const events: string[] = [];

    if (gameState.currentPhase !== lastPhaseRef.current) {
      if (lastPhaseRef.current !== null) {
        events.push(`Phase changed: ${lastPhaseRef.current} → ${gameState.currentPhase}`);
      } else {
        events.push(`Phase: ${gameState.currentPhase}`);
      }
      lastPhaseRef.current = gameState.currentPhase;
    }

    if (gameState.currentPlayer !== lastCurrentPlayerRef.current) {
      events.push(`Current player: P${gameState.currentPlayer}`);
      lastCurrentPlayerRef.current = gameState.currentPlayer;
    }

    if (pendingChoice && pendingChoice.id !== lastChoiceIdRef.current) {
      events.push(`Choice requested: ${pendingChoice.type} for P${pendingChoice.playerNumber}`);
      lastChoiceIdRef.current = pendingChoice.id;
    } else if (!pendingChoice && lastChoiceIdRef.current) {
      events.push('Choice resolved');
      lastChoiceIdRef.current = null;
    }

    if (events.length > 0) {
      setEventLog((prev) => {
        const next = [...events, ...prev];
        return next.slice(0, 50);
      });
    }
  }, [gameState, pendingChoice]);

  // Auto-resolved decision events
  useEffect(() => {
    if (!decisionAutoResolved) {
      return;
    }

    const { actingPlayerNumber, choiceKind, reason, resolvedMoveId } = decisionAutoResolved;
    const key = resolvedMoveId ?? `${actingPlayerNumber}:${choiceKind}:${reason}`;

    if (lastAutoResolvedKeyRef.current === key) {
      return;
    }

    lastAutoResolvedKeyRef.current = key;

    const label = describeDecisionAutoResolved(decisionAutoResolved);

    setEventLog((prev) => [label, ...prev].slice(0, 50));
  }, [decisionAutoResolved]);

  // Decision-phase timeout warning events
  useEffect(() => {
    if (!decisionPhaseTimeoutWarning) {
      return;
    }

    const { gameId, playerNumber, phase, remainingMs, choiceId } = decisionPhaseTimeoutWarning.data;

    const key = `${gameId}:${playerNumber}:${phase}:${choiceId ?? ''}:${remainingMs}`;
    if (lastTimeoutWarningKeyRef.current === key) {
      return;
    }
    lastTimeoutWarningKeyRef.current = key;

    const seconds = Math.max(1, Math.round(remainingMs / 1000));
    const label = `Decision timeout warning: P${playerNumber} in ${phase} (~${seconds}s remaining)`;

    setEventLog((prev) => [label, ...prev].slice(0, 50));
  }, [decisionPhaseTimeoutWarning]);

  // Connection status changes
  useEffect(() => {
    if (!connectionStatus || lastConnectionStatusRef.current === connectionStatus) {
      lastConnectionStatusRef.current = connectionStatus;
      return;
    }

    const label =
      connectionStatus === 'connected'
        ? 'Connection restored'
        : connectionStatus === 'reconnecting'
          ? 'Connection interrupted – reconnecting'
          : connectionStatus === 'connecting'
            ? 'Connecting to server…'
            : 'Disconnected from server';

    setEventLog((prev) => [label, ...prev].slice(0, 50));
    lastConnectionStatusRef.current = connectionStatus;
  }, [connectionStatus]);

  // Weird-state (ANM / forced elimination / structural stalemate / LPS) diagnostics.
  useEffect(() => {
    if (!gameState) {
      lastWeirdStateTypeRef.current = null;
      forcedElimContextRef.current = null;
      return;
    }

    const weird = getWeirdStateBanner(gameState, { victoryState });
    const prevType = lastWeirdStateTypeRef.current;
    const nextType = weird.type;

    const events: string[] = [];

    // Detect entry into ANM states (movement / line / territory).
    if (
      nextType === 'active-no-moves-movement' ||
      nextType === 'active-no-moves-line' ||
      nextType === 'active-no-moves-territory'
    ) {
      if (prevType !== nextType) {
        const phaseLabel =
          nextType === 'active-no-moves-movement'
            ? 'movement'
            : nextType === 'active-no-moves-line'
              ? 'line processing'
              : 'territory processing';
        const playerNumber = (weird as Extract<typeof weird, { playerNumber: number }>)
          .playerNumber;
        events.push(
          `Active–No–Moves detected for P${playerNumber} during ${phaseLabel}; the engine will apply forced resolution according to the rulebook.`
        );
      }
    }

    // Detect start and completion of forced elimination sequences.
    if (nextType === 'forced-elimination') {
      if (prevType !== 'forced-elimination') {
        const playerNumber = (weird as Extract<typeof weird, { playerNumber: number }>)
          .playerNumber;
        const startTotal =
          (gameState as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ?? 0;
        forcedElimContextRef.current = {
          active: true,
          startTotal,
          playerNumber,
        };
        events.push(`Forced elimination sequence started for P${playerNumber}.`);
      }
    } else if (prevType === 'forced-elimination') {
      const ctx = forcedElimContextRef.current;
      const endTotal =
        (gameState as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ?? 0;
      if (ctx) {
        const delta = Math.max(0, endTotal - ctx.startTotal);
        const playerLabel = ctx.playerNumber ? `P${ctx.playerNumber}` : 'the active player';
        const detail =
          delta > 0
            ? `${delta} ring${delta === 1 ? '' : 's'} were eliminated during the forced elimination sequence.`
            : 'No additional rings were eliminated during the forced elimination sequence.';
        events.push(`Forced elimination sequence completed for ${playerLabel}. ${detail}`);
      } else {
        events.push('Forced elimination sequence completed.');
      }
      forcedElimContextRef.current = null;
    }

    // Detect structural stalemate / plateau end conditions.
    if (nextType === 'structural-stalemate' && prevType !== 'structural-stalemate') {
      events.push(
        'Structural stalemate: no legal placements, movements, captures, or forced eliminations remain. The game ended by plateau auto-resolution.'
      );
    }

    // Detect last-player-standing terminal condition.
    if (nextType === 'last-player-standing' && prevType !== 'last-player-standing') {
      const winner = (weird as Extract<typeof weird, { winner?: number }>).winner;
      const label = typeof winner === 'number' ? `P${winner}` : 'A player';
      events.push(
        `Last Player Standing: ${label} won after three complete rounds where only they had real moves available.`
      );
    }

    if (events.length > 0) {
      setEventLog((prev) => {
        const merged = [...events, ...prev];
        return merged.slice(0, 50);
      });
    }

    lastWeirdStateTypeRef.current = nextType;
  }, [gameState, victoryState]);

  return {
    eventLog,
    showSystemEventsInLog,
    setShowSystemEventsInLog,
  };
}

/**
 * Backend DecisionUI hook: encapsulates pending choice state and countdown
 * timer wiring for the ChoiceDialog component.
 */
interface BackendDecisionUIState {
  pendingChoice: PlayerChoice | null;
  choiceDeadline: number | null;
  choiceTimeRemainingMs: number | null;
  respondToChoice: <T>(choice: PlayerChoice, selectedOption: T) => void;
  /**
   * Rich decision-phase view derived from choiceViewModels, used to provide
   * consistent copy/timeout semantics to both HUD and ChoiceDialog.
   */
  pendingChoiceView: PendingChoiceView | null;
}

function useBackendDecisionUI(): BackendDecisionUIState {
  const { choice, deadline, respond, timeRemaining, view } = usePendingChoice();

  return {
    pendingChoice: choice,
    choiceDeadline: deadline,
    choiceTimeRemainingMs: timeRemaining,
    // The underlying hook already knows which choice is pending; we ignore the
    // explicit choice argument and delegate to respond() for safety.
    respondToChoice: (_choice, selectedOption) => {
      respond(selectedOption as PlayerChoice['options'][number]);
    },
    pendingChoiceView: view,
  };
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

  // Selection + valid target highlighting
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);

  const [ringPlacementCountPrompt, setRingPlacementCountPrompt] = useState<{
    maxCount: number;
    hasStack: boolean;
    placeMovesAtPos: Move[];
  } | null>(null);

  // Diagnostics / host-level error banner
  const [fatalGameError, setFatalGameError] = useState<{
    message: string;
    technical?: string;
  } | null>(null);

  // Victory modal dismissal (backend only)
  const [isVictoryModalDismissed, setIsVictoryModalDismissed] = useState(false);

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

  // Resignation state
  const [isResigning, setIsResigning] = useState(false);
  const [isResignConfirmOpen, setIsResignConfirmOpen] = useState(false);

  // Weird-state tracking for rules-UX telemetry.
  // We only care about coarse weird-state categories (ANM/FE/structural stalemate/LPS)
  // and a single coarse timestamp for "seconds since weird state" on resign.
  const weirdStateFirstSeenAtRef = useRef<number | null>(null);
  const weirdStateTypeRef = useRef<RulesUxWeirdStateType | 'none'>('none');
  const weirdStateResignReportedRef = useRef<Set<string>>(new Set());

  // Difficulty calibration tracking: ensure we emit at most one
  // game_completed event per calibration game.
  const calibrationEventReportedRef = useRef(false);

  // Chat UI state (backend chat only; GameContext always provides sendChatMessage)
  const chatMessages = backendChatMessages;
  const [chatInput, setChatInput] = useState('');

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

  // Track the first time each weird-state type appears so that we can emit
  // coarse-grained rules-UX telemetry when the local player resigns while a
  // weird state is active.
  useEffect(() => {
    if (!gameState) {
      weirdStateTypeRef.current = 'none';
      weirdStateFirstSeenAtRef.current = null;
      return;
    }

    const weird = getWeirdStateBanner(gameState, { victoryState });
    const nextType = weird.type;
    const currentType = weirdStateTypeRef.current;

    if (nextType === 'none') {
      weirdStateTypeRef.current = 'none';
      weirdStateFirstSeenAtRef.current = null;
      return;
    }

    if (!isSurfaceableWeirdStateType(nextType as RulesUxWeirdStateType)) {
      weirdStateTypeRef.current = 'none';
      weirdStateFirstSeenAtRef.current = null;
      return;
    }

    if (currentType !== nextType) {
      weirdStateTypeRef.current = nextType as RulesUxWeirdStateType;
      weirdStateFirstSeenAtRef.current = Date.now();
    }
  }, [gameState, victoryState]);

  // Emit a single difficulty_calibration_game_completed event for calibration games
  // that have a stored session (created from the Lobby). This remains purely
  // client-driven for now and does not depend on server-side calibration flags.
  useEffect(() => {
    if (!victoryState || calibrationEventReportedRef.current) {
      return;
    }

    const session = getDifficultyCalibrationSession(routeGameId);
    if (!session || !session.isCalibrationOptIn) {
      return;
    }

    if (!gameState) {
      return;
    }

    let result: 'win' | 'loss' | 'draw' | 'abandoned' = 'abandoned';

    if (victoryState.reason === 'draw') {
      result = 'draw';
    } else if (victoryState.reason === 'abandonment') {
      result = 'abandoned';
    } else if (typeof victoryState.winner === 'number') {
      const winnerPlayer = gameState.players.find((p) => p.playerNumber === victoryState.winner);
      if (winnerPlayer?.type === 'human') {
        result = 'win';
      } else if (winnerPlayer?.type === 'ai') {
        result = 'loss';
      }
    }

    const movesPlayed =
      Array.isArray(gameState.moveHistory) && gameState.moveHistory.length > 0
        ? gameState.moveHistory.length
        : undefined;

    calibrationEventReportedRef.current = true;
    clearDifficultyCalibrationSession(routeGameId);

    void sendDifficultyCalibrationEvent({
      type: 'difficulty_calibration_game_completed',
      boardType: session.boardType,
      numPlayers: session.numPlayers,
      difficulty: session.difficulty,
      isCalibrationOptIn: true,
      result,
      movesPlayed,
    });
  }, [victoryState, gameState, routeGameId]);

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
  }, [routeGameId]);

  // Reset backend victory modal dismissal whenever the active game or victory state changes
  useEffect(() => {
    setIsVictoryModalDismissed(false);
  }, [routeGameId, victoryState]);

  // When a rematch has been accepted and the server provides a new gameId,
  // navigate to the fresh backend game route. The GameProvider connection
  // shell will tear down the old WebSocket and establish a new one based
  // on the updated :gameId route param.
  useEffect(() => {
    if (rematchGameId) {
      navigate(`/game/${rematchGameId}`);
    }
  }, [rematchGameId, navigate]);

  // Auto-highlight valid placement targets during ring_placement
  useEffect(() => {
    if (!gameState) return;

    if (gameState.currentPhase === 'ring_placement') {
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const placementTargets = validMoves.filter((m) => m.type === 'place_ring').map((m) => m.to);

        setValidTargets((prev) => {
          if (prev.length !== placementTargets.length) return placementTargets;
          const allMatch = prev.every((p) => placementTargets.some((pt) => positionsEqual(p, pt)));
          return allMatch ? prev : placementTargets;
        });
      } else {
        // Only clear if not already empty, avoiding unnecessary re-renders
        setValidTargets((prev) => (prev.length === 0 ? prev : []));
      }
    }
  }, [gameState?.currentPhase, validMoves]);

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

  // Approximate must-move stack highlighting: if all movement/capture moves
  // originate from the same stack, treat that as the forced origin.
  const backendMustMoveFrom: Position | undefined = useMemo(() => {
    if (!Array.isArray(validMoves) || !gameState) return undefined;
    if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
      return undefined;
    }

    const origins = validMoves
      .filter((m) => m.from && (m.type === 'move_stack' || m.type === 'overtaking_capture'))
      .map((m) => m.from as Position);

    if (origins.length === 0) return undefined;
    const first = origins[0];
    const allSame = origins.every((p) => positionsEqual(p, first));
    return allSame ? first : undefined;
  }, [validMoves, gameState]);

  // Extract chain capture path for visualisation during chain_capture phase.
  // The path includes the starting position and all landing positions visited
  // in order, mirroring the sandbox host semantics so overlays remain
  // consistent between backend and local games.
  const chainCapturePath: Position[] | undefined = useMemo(() => {
    if (!gameState || gameState.currentPhase !== 'chain_capture') {
      return undefined;
    }

    const moveHistory = gameState.moveHistory;
    if (!moveHistory || moveHistory.length === 0) {
      return undefined;
    }

    const currentPlayerNumber = gameState.currentPlayer;
    const path: Position[] = [];

    for (let i = moveHistory.length - 1; i >= 0; i--) {
      const move = moveHistory[i];
      if (!move) continue;

      if (
        move.player !== currentPlayerNumber ||
        (move.type !== 'overtaking_capture' && move.type !== 'continue_capture_segment')
      ) {
        break;
      }

      if (move.to) {
        path.unshift(move.to);
      }

      if (move.type === 'overtaking_capture' && move.from) {
        path.unshift(move.from);
      }
    }

    return path.length >= 2 ? path : undefined;
  }, [gameState]);

  // Backend board click handling
  const handleBackendCellClick = (pos: Position, board: BoardState) => {
    if (!gameState) return;
    const posKey = positionToString(pos);

    if (!isPlayer) {
      toast.error('Spectators cannot submit moves', { id: 'interaction-locked' });
      return;
    }

    if (!isConnectionActive) {
      toast.error('Moves paused while disconnected', { id: 'interaction-locked' });
      return;
    }

    // Ring placement phase: attempt canonical 1-ring placement on empties
    if (gameState.currentPhase === 'ring_placement') {
      if (!Array.isArray(validMoves) || validMoves.length === 0) {
        return;
      }

      const hasStack = !!board.stacks.get(posKey);

      if (!hasStack) {
        const placeMovesAtPos = validMoves.filter(
          (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
        );
        if (placeMovesAtPos.length === 0) {
          // Use enhanced invalid move feedback with shake animation and explanatory toast
          const reason = analyzeInvalid(gameState, pos, {
            isPlayer,
            isMyTurn,
            isConnected: isConnectionActive,
            selectedPosition: selected,
            validMoves: validMoves ?? undefined,
            mustMoveFrom: backendMustMoveFrom,
          });
          triggerInvalidMove(pos, reason);
          return;
        }

        const preferred =
          placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];

        submitMove({
          type: 'place_ring',
          to: preferred.to,
          placementCount: preferred.placementCount,
          placedOnStack: preferred.placedOnStack,
        } as PartialMove);

        setSelected(undefined);
        setValidTargets([]);
        return;
      }

      // Clicking stacks in placement phase just selects them.
      setSelected(pos);
      setValidTargets([]);
      return;
    }

    // Movement/capture phases: select source, then target.
    if (!selected) {
      // When there are no valid moves at all, keep the previous behaviour and
      // simply allow selection without special feedback.
      if (!Array.isArray(validMoves) || validMoves.length === 0) {
        setSelected(pos);
        setValidTargets([]);
        return;
      }

      const hasStack = !!board.stacks.get(posKey);
      const hasMovesFromHere = validMoves.some(
        (m) => m.from && positionsEqual(m.from as Position, pos)
      );

      if (hasStack && hasMovesFromHere) {
        setSelected(pos);
        const targets = validMoves
          .filter((m) => m.from && positionsEqual(m.from as Position, pos))
          .map((m) => m.to);
        setValidTargets(targets);
      } else {
        const reason = analyzeInvalid(gameState, pos, {
          isPlayer,
          isMyTurn,
          isConnected: isConnectionActive,
          selectedPosition: null,
          validMoves: validMoves ?? undefined,
          mustMoveFrom: backendMustMoveFrom,
        });
        triggerInvalidMove(pos, reason);
      }
      return;
    }

    // Clicking the same cell clears selection.
    if (positionsEqual(selected, pos)) {
      setSelected(undefined);
      setValidTargets([]);
      return;
    }

    // If highlighted and a matching move exists, submit.
    if (Array.isArray(validMoves) && validMoves.length > 0) {
      const matching = validMoves.find(
        (m) => m.from && positionsEqual(m.from, selected) && positionsEqual(m.to, pos)
      );

      if (matching) {
        submitMove({
          type: matching.type,
          from: matching.from,
          to: matching.to,
        } as PartialMove);

        setSelected(undefined);
        setValidTargets([]);
        return;
      }
    }

    // Otherwise, treat either as a new (valid) selection or as an invalid
    // landing/selection and surface feedback.
    const hasStack = !!board.stacks.get(posKey);
    const hasMovesFromHere =
      Array.isArray(validMoves) &&
      validMoves.some((m) => m.from && positionsEqual(m.from as Position, pos));

    if (hasStack && hasMovesFromHere) {
      setSelected(pos);
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const targets = validMoves
          .filter((m) => m.from && positionsEqual(m.from as Position, pos))
          .map((m) => m.to);
        setValidTargets(targets);
      } else {
        setValidTargets([]);
      }
    } else {
      const reason = analyzeInvalid(gameState, pos, {
        isPlayer,
        isMyTurn,
        isConnected: isConnectionActive,
        selectedPosition: selected ?? null,
        validMoves: validMoves ?? undefined,
        mustMoveFrom: backendMustMoveFrom,
      });
      triggerInvalidMove(pos, reason);
    }
  };

  // Backend double-click handling for ring_placement (prefer 2-ring where legal)
  const handleBackendCellDoubleClick = (pos: Position, board: BoardState) => {
    if (!gameState) return;
    if (!isPlayer || !isConnectionActive) {
      toast.error('Cannot modify placements while disconnected or spectating', {
        id: 'interaction-locked',
      });
      return;
    }
    if (gameState.currentPhase !== 'ring_placement') {
      return;
    }

    if (!Array.isArray(validMoves) || validMoves.length === 0) {
      return;
    }

    const key = positionToString(pos);
    const hasStack = !!board.stacks.get(key);

    const placeMovesAtPos = validMoves.filter(
      (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
    );
    if (placeMovesAtPos.length === 0) {
      return;
    }

    let chosen: Move | undefined;

    if (!hasStack) {
      const twoRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 2);
      const oneRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1);
      chosen = twoRing || oneRing || placeMovesAtPos[0];
    } else {
      chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];
    }

    if (!chosen) {
      return;
    }

    submitMove({
      type: 'place_ring',
      to: chosen.to,
      placementCount: chosen.placementCount,
      placedOnStack: chosen.placedOnStack,
    } as PartialMove);

    setSelected(undefined);
    setValidTargets([]);
  };

  // Backend context-menu placement handler
  const handleBackendCellContextMenu = (pos: Position, board: BoardState) => {
    if (!gameState) return;
    if (!isPlayer || !isConnectionActive) {
      toast.error('Cannot modify placements while disconnected or spectating', {
        id: 'interaction-locked',
      });
      return;
    }
    if (gameState.currentPhase !== 'ring_placement') {
      return;
    }

    if (!Array.isArray(validMoves) || validMoves.length === 0) {
      return;
    }

    const key = positionToString(pos);
    const hasStack = !!board.stacks.get(key);

    const placeMovesAtPos = validMoves.filter(
      (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
    );
    if (placeMovesAtPos.length === 0) {
      return;
    }

    const counts = placeMovesAtPos.map((m) => m.placementCount ?? 1);
    const maxCount = Math.max(...counts);

    if (maxCount <= 1) {
      const chosen =
        placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];
      if (!chosen) return;

      submitMove({
        type: 'place_ring',
        to: chosen.to,
        placementCount: chosen.placementCount,
        placedOnStack: chosen.placedOnStack,
      } as PartialMove);

      setSelected(undefined);
      setValidTargets([]);
      return;
    }

    setRingPlacementCountPrompt({
      maxCount,
      hasStack,
      placeMovesAtPos,
    });
  };

  const handleConfirmRingPlacementCount = (count: number) => {
    const prompt = ringPlacementCountPrompt;
    if (!prompt) return;

    const chosen = prompt.placeMovesAtPos.find((m) => (m.placementCount ?? 1) === count);
    if (!chosen) {
      setRingPlacementCountPrompt(null);
      return;
    }

    submitMove({
      type: 'place_ring',
      to: chosen.to,
      placementCount: chosen.placementCount,
      placedOnStack: chosen.placedOnStack,
    } as PartialMove);

    setSelected(undefined);
    setValidTargets([]);
    setRingPlacementCountPrompt(null);
  };

  // Handle resignation
  const handleResign = async () => {
    if (!gameId || isResigning) return;

    setIsResigning(true);
    try {
      const weirdType = weirdStateTypeRef.current;
      const firstSeenAt = weirdStateFirstSeenAtRef.current;

      // Derive coarse board / difficulty context from the current GameState.
      let boardTypeForTelemetry: GameState['boardType'] | undefined;
      let numPlayersForTelemetry: number | undefined;
      let aiDifficultyForTelemetry: number | undefined;

      if (gameState) {
        boardTypeForTelemetry = gameState.boardType;
        numPlayersForTelemetry = gameState.players.length;

        const aiPlayers = gameState.players.filter((p) => p.type === 'ai');
        let maxDifficulty = 0;
        for (const p of aiPlayers) {
          const d = p.aiProfile?.difficulty ?? p.aiDifficulty;
          if (typeof d === 'number' && Number.isFinite(d) && d > maxDifficulty) {
            maxDifficulty = d;
          }
        }
        if (maxDifficulty > 0) {
          aiDifficultyForTelemetry = maxDifficulty;
        }
      }

      if (
        boardTypeForTelemetry &&
        typeof numPlayersForTelemetry === 'number' &&
        numPlayersForTelemetry > 0 &&
        weirdType &&
        weirdType !== 'none' &&
        !weirdStateResignReportedRef.current.has(weirdType)
      ) {
        const secondsSinceWeirdState =
          typeof firstSeenAt === 'number'
            ? Math.max(0, Math.round((Date.now() - firstSeenAt) / 1000))
            : undefined;

        weirdStateResignReportedRef.current.add(weirdType);

        void sendRulesUxEvent({
          type: 'rules_weird_state_resign',
          boardType: boardTypeForTelemetry,
          numPlayers: numPlayersForTelemetry,
          aiDifficulty: aiDifficultyForTelemetry,
          weirdStateType: weirdType as RulesUxWeirdStateType,
          secondsSinceWeirdState,
        });
      }

      await gameApi.leaveGame(gameId);
      toast.success('You have resigned from the game.');
      // The server will broadcast victory/game over state via WebSocket
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to resign';
      toast.error(message);
      setIsResigning(false);
    }
  };

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
        onClose={() => {
          setIsVictoryModalDismissed(true);
        }}
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
        <section className="flex-shrink-0 flex justify-center lg:justify-start">
          <BoardView
            boardType={boardType}
            board={board}
            viewModel={backendBoardViewModel}
            selectedPosition={selected || backendMustMoveFrom}
            validTargets={validTargets}
            onCellClick={(pos) => handleBackendCellClick(pos, board)}
            onCellDoubleClick={(pos) => handleBackendCellDoubleClick(pos, board)}
            onCellContextMenu={(pos) => handleBackendCellContextMenu(pos, board)}
            isSpectator={!isPlayer}
            pendingAnimation={pendingAnimation ?? undefined}
            onAnimationComplete={clearAnimation}
            chainCapturePath={chainCapturePath}
            shakingCellKey={shakingCellKey}
            onShowKeyboardHelp={() => setShowBoardControls(true)}
          />
        </section>

        <aside className="w-full lg:w-80 flex-shrink-0 space-y-3 text-sm text-slate-100">
          {/* Primary HUD band – placed at the top of the sidebar so phase/turn/time
              are always visible alongside the board. On mobile, render the
              compact MobileGameHUD; on larger screens, use the full GameHUD. */}
          {isMobile ? (
            <MobileGameHUD
              viewModel={hudViewModel}
              timeControl={gameState.timeControl}
              onShowBoardControls={() => setShowBoardControls(true)}
              rulesUxContext={rulesUxContext}
            />
          ) : (
            <GameHUD
              viewModel={hudViewModel}
              timeControl={gameState.timeControl}
              onShowBoardControls={() => setShowBoardControls(true)}
              rulesUxContext={rulesUxContext}
            />
          )}

          {isPlayer && (
            <ChoiceDialog
              choice={pendingChoice}
              choiceViewModel={pendingChoiceView?.viewModel}
              deadline={choiceDeadline}
              timeRemainingMs={reconciledDecisionTimeRemainingMs}
              isServerCapped={decisionCountdown.isServerCapped}
              onSelectOption={(choice, option) => respondToChoice(choice, option)}
            />
          )}

          <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
            <h2 className="font-semibold mb-2">Selection</h2>
            {selected ? (
              <div className="space-y-2">
                <div className="text-lg font-mono font-semibold text-white">
                  {formatPosition(selected, { boardType })}
                </div>
                {backendSelectedStackDetails ? (
                  <ul className="text-xs text-slate-300 space-y-1">
                    <li>Stack height: {backendSelectedStackDetails.height}</li>
                    <li>Cap height: {backendSelectedStackDetails.cap}</li>
                    <li>Controlled by: P{backendSelectedStackDetails.controllingPlayer}</li>
                  </ul>
                ) : (
                  <p className="text-xs text-slate-300">Empty cell – choose a placement target.</p>
                )}
                <p className="text-xs text-slate-400">
                  Click a highlighted destination to commit the move, or select a new source.
                </p>
              </div>
            ) : (
              <div className="text-slate-200">Click a cell to inspect it.</div>
            )}
            {boardInteractionMessage && (
              <div className="mt-3 text-xs text-amber-300">{boardInteractionMessage}</div>
            )}
          </div>

          {/* Move History - compact notation display */}
          <MoveHistory
            moves={gameState.moveHistory}
            boardType={gameState.boardType}
            currentMoveIndex={gameState.moveHistory.length - 1}
          />

          {/* Resign button - visible to players during active games */}
          {isPlayer && gameState.gameStatus === 'active' && (
            <div className="mt-2 flex justify-end">
              <ResignButton
                onResign={handleResign}
                disabled={!isConnectionActive}
                isResigning={isResigning}
                isConfirmOpen={isResignConfirmOpen}
                onConfirmOpenChange={setIsResignConfirmOpen}
              />
            </div>
          )}

          {isPlayer &&
            isConnectionActive &&
            gameState.gameStatus === 'active' &&
            gameState.players.length === 2 &&
            gameState.rulesOptions?.swapRuleEnabled === true &&
            hudCurrentPlayer &&
            hudCurrentPlayer.playerNumber === gameState.currentPlayer &&
            hudCurrentPlayer.playerNumber === 2 &&
            // One-time, immediately after Player 1's first turn:
            !gameState.moveHistory.some((m) => m.type === 'swap_sides') &&
            gameState.moveHistory.some((m) => m.player === 1) &&
            !gameState.moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides') && (
              <div className="mt-2 p-2 border border-amber-500/60 rounded bg-amber-900/40 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-amber-100">
                    Pie rule available: swap colours with Player 1.
                  </span>
                  <button
                    type="button"
                    className="px-2 py-1 rounded bg-amber-500 hover:bg-amber-400 text-black font-semibold"
                    onClick={() => {
                      submitMove({
                        type: 'swap_sides',
                        to: { x: 0, y: 0 },
                      } as PartialMove);
                    }}
                  >
                    Swap colours
                  </button>
                </div>
                <p className="mt-1 text-amber-100/80">
                  As Player 2, you may use this once, immediately after Player 1’s first turn.
                </p>
              </div>
            )}

          {decisionAutoResolved && (
            <div className="mt-1 text-[11px] text-amber-300">
              {describeDecisionAutoResolved(decisionAutoResolved)}
            </div>
          )}

          <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1">
            <span>Log view</span>
            <button
              type="button"
              onClick={() => setShowSystemEventsInLog((prev) => !prev)}
              className="px-2 py-0.5 rounded border border-slate-600 bg-slate-900/70 text-xs hover:border-emerald-400 hover:text-emerald-200 transition"
            >
              {showSystemEventsInLog ? 'Moves + system' : 'Moves only'}
            </button>
          </div>

          <GameEventLog
            viewModel={toEventLogViewModel(
              gameState.history,
              showSystemEventsInLog ? eventLog : [],
              victoryState,
              { maxEntries: 40 }
            )}
          />

          <details
            className="p-3 border border-slate-700 rounded bg-slate-900/50"
            open={showAdvancedSidebarPanels}
            onToggle={(event) => {
              setShowAdvancedSidebarPanels(event.currentTarget.open);
            }}
            data-testid="backend-advanced-sidebar-panels"
          >
            <summary className="cursor-pointer select-none text-sm font-semibold text-slate-200">
              Advanced diagnostics
              <span className="ml-2 text-[11px] font-normal text-slate-400">
                (history, evaluation)
              </span>
            </summary>
            {showAdvancedSidebarPanels && (
              <div className="mt-3 space-y-3">
                {/* Full move history panel with expandable details */}
                <GameHistoryPanel
                  gameId={gameId}
                  defaultCollapsed={true}
                  onError={(err) => {
                    // Log but don't block – history is supplementary
                    console.warn('Failed to load game history:', err.message);
                  }}
                />

                {/* AI analysis/evaluation panel – enabled for spectators and finished games.
                    When no evaluation data has been streamed yet, the panel renders a
                    placeholder message instead of remaining hidden. */}
                {(!isPlayer || !!victoryState) && gameState && (
                  <EvaluationPanel
                    evaluationHistory={evaluationHistory}
                    players={gameState.players}
                    className="mt-2"
                  />
                )}
              </div>
            )}
          </details>

          <div className="p-3 border border-slate-700 rounded bg-slate-900/50 flex flex-col h-64">
            <h2 className="font-semibold mb-2">Chat</h2>
            <div className="flex-1 overflow-y-auto mb-2 space-y-1">
              {chatMessages.length === 0 ? (
                <div className="text-slate-400 text-xs italic">No messages yet.</div>
              ) : (
                chatMessages.map((msg, idx) => (
                  <div key={idx} className="text-xs">
                    <span className="font-bold text-slate-300">{msg.sender}: </span>
                    <span className="text-slate-200">{msg.text}</span>
                  </div>
                ))
              )}
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!chatInput.trim()) return;

                sendChatMessage(chatInput);
                setChatInput('');
              }}
              className="flex gap-2"
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Type a message..."
                aria-label="Chat message"
                className="flex-1 bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-emerald-500"
              />
              <Button type="submit" size="sm">
                Send
              </Button>
            </form>
          </div>
        </aside>
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
        onClose={() => setRingPlacementCountPrompt(null)}
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
