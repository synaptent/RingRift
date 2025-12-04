import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameHUD } from '../components/GameHUD';
import { GameEventLog } from '../components/GameEventLog';
import { MoveHistory } from '../components/MoveHistory';
import { GameHistoryPanel } from '../components/GameHistoryPanel';
import { EvaluationPanel } from '../components/EvaluationPanel';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import { ResignButton } from '../components/ResignButton';
import {
  ScreenReaderAnnouncer,
  useScreenReaderAnnouncement,
} from '../components/ScreenReaderAnnouncer';
import { gameApi } from '../services/api';
import {
  BoardState,
  GameState,
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
import { getGameOverBannerText } from '../utils/gameCopy';
import { useGameState } from '../hooks/useGameState';
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
      <h1 className="text-3xl font-bold mb-1">Game</h1>
      <p className="text-sm text-gray-500">
        Game ID: {gameState.id} • Board: {gameState.boardType} • Players: {playerSummary}
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
  decisionPhaseTimeoutWarning: DecisionPhaseTimeoutWarningPayload | null
): BackendDiagnosticsState {
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [showSystemEventsInLog, setShowSystemEventsInLog] = useState(true);

  const lastPhaseRef = useRef<string | null>(null);
  const lastCurrentPlayerRef = useRef<number | null>(null);
  const lastChoiceIdRef = useRef<string | null>(null);
  const lastAutoResolvedKeyRef = useRef<string | null>(null);
  const lastConnectionStatusRef = useRef<ConnectionStatus | null>(null);
  const lastTimeoutWarningKeyRef = useRef<string | null>(null);

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
  const { pendingAnimation, clearAnimation } = useAutoMoveAnimation(gameState);

  // Invalid move feedback - shake animation and explanatory toasts
  const { shakingCellKey, triggerInvalidMove } = useInvalidMoveFeedback();

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
    decisionPhaseTimeoutWarning
  );

  const connectionStatus = connection.connectionStatus;
  const { isConnecting, error, lastHeartbeatAt } = connection;

  // Selection + valid target highlighting
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);

  // Diagnostics / host-level error banner
  const [fatalGameError, setFatalGameError] = useState<{
    message: string;
    technical?: string;
  } | null>(null);

  // Victory modal dismissal (backend only)
  const [isVictoryModalDismissed, setIsVictoryModalDismissed] = useState(false);

  // Help / controls overlay
  const [showBoardControls, setShowBoardControls] = useState(false);

  // Resignation state
  const [isResigning, setIsResigning] = useState(false);

  // Chat UI state (backend chat only; GameContext always provides sendChatMessage)
  const chatMessages = backendChatMessages;
  const [chatInput, setChatInput] = useState('');

  // Screen reader announcements for accessibility
  const { message: srMessage, announce: srAnnounce } = useScreenReaderAnnouncement();
  const prevTurnPlayerRef = useRef<number | null>(null);
  const prevPhaseRef = useRef<string | null>(null);
  const prevVictoryRef = useRef<boolean>(false);

  // Announce turn and phase changes to screen readers
  useEffect(() => {
    if (!gameState) return;

    const currentTurnPlayer = gameState.currentPlayer;
    const currentPhase = gameState.currentPhase;
    const currentPlayerInfo = gameState.players.find((p) => p.playerNumber === currentTurnPlayer);
    const playerName = currentPlayerInfo?.username || `Player ${currentTurnPlayer}`;

    // Announce turn changes
    if (prevTurnPlayerRef.current !== null && prevTurnPlayerRef.current !== currentTurnPlayer) {
      srAnnounce(`${playerName}'s turn`);
    }
    prevTurnPlayerRef.current = currentTurnPlayer;

    // Announce significant phase changes (skip if also announcing turn)
    if (prevPhaseRef.current !== null && prevPhaseRef.current !== currentPhase) {
      const phaseLabels: Record<string, string> = {
        ring_placement: 'Ring placement phase',
        movement: 'Movement phase',
        capture: 'Capture phase',
        chain_capture: 'Chain capture phase',
        line_processing: 'Line processing phase',
        territory_processing: 'Territory processing phase',
      };
      const phaseLabel = phaseLabels[currentPhase] || currentPhase;
      // Only announce phase if it wasn't just a turn change announcement
      if (prevTurnPlayerRef.current === currentTurnPlayer) {
        srAnnounce(phaseLabel);
      }
    }
    prevPhaseRef.current = currentPhase;
  }, [gameState, srAnnounce]);

  // Announce victory
  useEffect(() => {
    if (victoryState && !prevVictoryRef.current) {
      const winnerInfo =
        victoryState.winner !== undefined
          ? gameState?.players.find((p) => p.playerNumber === victoryState.winner)
          : null;
      const winnerName =
        winnerInfo?.username ||
        (victoryState.winner !== undefined ? `Player ${victoryState.winner}` : '');

      let announcement: string;

      if (victoryState.winner === undefined) {
        // No explicit winner – draw or abandoned game.
        switch (victoryState.reason) {
          case 'draw':
            announcement = 'Game over. The game ended in a draw.';
            break;
          case 'abandonment':
            announcement = 'Game over. The game was abandoned.';
            break;
          default:
            announcement = 'Game over.';
            break;
        }
      } else {
        let reasonLabel = '';
        switch (victoryState.reason) {
          case 'ring_elimination':
            reasonLabel = 'by elimination';
            break;
          case 'territory_control':
            reasonLabel = 'by territory control';
            break;
          case 'last_player_standing':
            reasonLabel = 'as the last player standing';
            break;
          case 'timeout':
            reasonLabel = 'on time';
            break;
          case 'resignation':
            reasonLabel = 'by resignation';
            break;
          case 'abandonment':
            reasonLabel = 'after the game was abandoned';
            break;
          default:
            reasonLabel = '';
            break;
        }

        const suffix = reasonLabel ? ` ${reasonLabel}` : '';
        announcement = `Game over. ${winnerName} wins${suffix}.`;
      }

      srAnnounce(announcement.trim());
      prevVictoryRef.current = true;
    } else if (!victoryState) {
      prevVictoryRef.current = false;
    }
  }, [victoryState, gameState?.players, srAnnounce]);

  // Derived HUD state
  const currentPlayer = gameState?.players.find((p) => p.playerNumber === gameState.currentPlayer);
  const isPlayer = !!gameState?.players.some((p) => p.id === user?.id);
  const isMyTurn = currentPlayer?.id === user?.id;
  const isConnectionActive = connectionStatus === 'connected';
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
  }, [routeGameId]);

  // Reset backend victory modal dismissal whenever the active game or victory state changes
  useEffect(() => {
    setIsVictoryModalDismissed(false);
  }, [routeGameId, victoryState]);

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
        setValidTargets([]);
      }
    }
  }, [gameState?.currentPhase, validMoves]);

  // Diagnostics for phase / choice / connection are handled by useBackendDiagnosticsLog.
  // Choice countdown is owned by useBackendDecisionUI.
  // Global keyboard shortcuts (desktop): "?" toggles the board controls
  // overlay; Escape closes it when open. Kept at the host layer so that
  // presentation components remain rules-agnostic.
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
  }, [showBoardControls]);

  // Approximate must-move stack highlighting: if all movement/capture moves
  // originate from the same stack, treat that as the forced origin.
  const backendMustMoveFrom: Position | undefined = useMemo(() => {
    if (!Array.isArray(validMoves) || !gameState) return undefined;
    if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
      return undefined;
    }

    const origins = validMoves
      .filter(
        (m) =>
          m.from &&
          (m.type === 'move_stack' ||
            m.type === 'move_ring' ||
            m.type === 'build_stack' ||
            m.type === 'overtaking_capture')
      )
      .map((m) => m.from as Position);

    if (origins.length === 0) return undefined;
    const first = origins[0];
    const allSame = origins.every((p) => positionsEqual(p, first));
    return allSame ? first : undefined;
  }, [validMoves, gameState]);

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
            validMoves,
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
          validMoves,
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
        validMoves,
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

    const promptLabel = hasStack
      ? 'Place how many rings on this stack? (canonical: 1)'
      : `Place how many rings on this empty cell? (1–${maxCount})`;

    const raw = window.prompt(promptLabel, Math.min(2, maxCount).toString());
    if (!raw) {
      return;
    }

    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > maxCount) {
      return;
    }

    const chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === parsed);
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

  // Handle resignation
  const handleResign = async () => {
    if (!gameId || isResigning) return;

    setIsResigning(true);
    try {
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
      <div className="bg-amber-500/20 border border-amber-500/50 text-amber-200 px-4 py-2 rounded mb-4 flex items-center justify-between">
        <span>
          {connectionStatus === 'reconnecting'
            ? 'Connection lost. Attempting to reconnect…'
            : connectionStatus === 'connecting'
              ? 'Connecting to game server…'
              : 'Disconnected from server. Moves are paused.'}
        </span>
        <div className="animate-spin h-4 w-4 border-2 border-amber-500 border-t-transparent rounded-full" />
      </div>
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

  // Derive decision-phase board highlights (if any) from the current GameState
  // and pending PlayerChoice, then feed them into the BoardViewModel so the
  // BoardView can render consistent geometry overlays for all roles.
  const decisionHighlights = deriveBoardDecisionHighlights(gameState, pendingChoice);

  const backendBoardViewModel = toBoardViewModel(board, {
    selectedPosition: selected || backendMustMoveFrom,
    validTargets,
    decisionHighlights,
  });

  const hudViewModel = toHUDViewModel(gameState, {
    instruction: getInstruction(),
    connectionStatus,
    lastHeartbeatAt,
    isSpectator: !isPlayer,
    currentUserId: user?.id,
    pendingChoice,
    choiceDeadline,
    choiceTimeRemainingMs: reconciledDecisionTimeRemainingMs,
    decisionIsServerCapped: decisionCountdown.isServerCapped,
  });

  const victoryViewModel = toVictoryViewModel(victoryState, gameState.players, gameState, {
    currentUserId: user?.id,
    isDismissed: isVictoryModalDismissed,
  });

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
    <div className="container mx-auto px-4 py-8 space-y-4">
      {/* Screen reader live region for game announcements */}
      <ScreenReaderAnnouncer message={srMessage} />

      {reconnectionBanner}

      {/* Opponent disconnection banner - shown when opponents have disconnected but game continues */}
      {disconnectedOpponents.length > 0 && !gameEndedByAbandonment && !victoryState && (
        <div className="bg-orange-500/20 border border-orange-500/50 text-orange-200 px-4 py-2 rounded mb-4 flex items-center gap-3">
          <svg
            className="w-5 h-5 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <span>
            {disconnectedOpponents.length === 1
              ? `${disconnectedOpponents[0].username || 'A player'} has disconnected. Waiting for reconnection…`
              : `${disconnectedOpponents.length} players have disconnected. Waiting for reconnection…`}
          </span>
          <div className="ml-auto animate-pulse h-2 w-2 rounded-full bg-orange-400" />
        </div>
      )}

      {/* Abandonment banner - shown when game ended due to reconnection timeout */}
      {gameEndedByAbandonment && (
        <div className="bg-red-500/20 border border-red-500/50 text-red-200 px-4 py-2 rounded mb-4">
          <span className="font-semibold">Game ended by abandonment.</span>
          <span className="ml-2 text-red-300/80">
            A player failed to reconnect within the allowed time.
          </span>
        </div>
      )}

      {gameOverBannerText && (
        <div className="bg-emerald-900/30 border border-emerald-500/60 text-emerald-100 px-4 py-2 rounded mb-2 text-sm">
          {gameOverBannerText}
        </div>
      )}

      {fatalGameError && (
        <div className="bg-red-500/20 border border-red-500/50 text-red-200 px-4 py-3 rounded">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="font-semibold mb-1">{fatalGameError.message}</p>
              {process.env.NODE_ENV === 'development' && fatalGameError.technical && (
                <p className="text-xs text-red-300 font-mono mt-2">
                  Technical: {fatalGameError.technical}
                </p>
              )}
            </div>
            <button
              onClick={() => setFatalGameError(null)}
              className="ml-4 text-red-300 hover:text-red-100 transition"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <header className="flex items-center justify-between">
        <div>
          {renderGameHeader(gameState)}
          {!isPlayer && (
            <span className="ml-2 px-2 py-0.5 bg-purple-900/50 border border-purple-500/50 text-purple-200 text-xs rounded-full uppercase tracking-wider font-bold">
              Spectating
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2 text-xs text-gray-400">
          <span>Status: {gameState.gameStatus}</span>
          <span>• Phase: {hudViewModel.phase.label}</span>
          <span>
            • Current player:{' '}
            {hudCurrentPlayer
              ? hudCurrentPlayer.username || `P${hudCurrentPlayer.playerNumber}`
              : '—'}
          </span>
        </div>
      </header>

      <VictoryModal
        isOpen={!!victoryState && !isVictoryModalDismissed}
        viewModel={victoryViewModel}
        onClose={() => {
          setIsVictoryModalDismissed(true);
        }}
        onReturnToLobby={() => navigate('/lobby')}
        onRematch={async () => {
          // Create a new game with the same settings as the current one
          if (!gameState) {
            toast.error('Cannot create rematch: game state unavailable');
            return;
          }

          try {
            const newGame = await gameApi.createGame({
              boardType: gameState.boardType,
              maxPlayers: gameState.players.length,
              isRated: false,
              isPrivate: true,
              timeControl: gameState.timeControl ?? {
                type: 'rapid',
                initialTime: 600,
                increment: 0,
              },
              aiOpponents: (() => {
                const aiPlayers = gameState.players.filter((p) => p.type === 'ai');
                if (aiPlayers.length === 0) return undefined;
                return {
                  count: aiPlayers.length,
                  difficulty: aiPlayers.map((p) => p.aiProfile?.difficulty ?? p.aiDifficulty ?? 5),
                  mode: 'service' as const,
                  aiType: 'heuristic' as const,
                };
              })(),
              rulesOptions: gameState.rulesOptions,
            });
            toast.success('Rematch game created!');
            navigate(`/game/${newGame.id}`);
          } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to create rematch';
            toast.error(message);
          }
        }}
      />

      <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
        <section>
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
            shakingCellKey={shakingCellKey}
          />
        </section>

        <aside className="w-full md:w-72 space-y-3 text-sm text-slate-100">
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
                  ({selected.x}, {selected.y}
                  {selected.z !== undefined ? `, ${selected.z}` : ''})
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

          <GameHUD
            viewModel={hudViewModel}
            timeControl={gameState.timeControl}
            onShowBoardControls={() => setShowBoardControls(true)}
          />

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
                className="flex-1 bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-emerald-500"
              />
              <button
                type="submit"
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 rounded text-xs font-medium"
              >
                Send
              </button>
            </form>
          </div>
        </aside>
      </main>

      {showBoardControls && (
        <BoardControlsOverlay
          mode={isPlayer ? 'backend' : 'spectator'}
          onClose={() => setShowBoardControls(false)}
        />
      )}
    </div>
  );
};
