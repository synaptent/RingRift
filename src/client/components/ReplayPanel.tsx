import { useState, useEffect, useCallback, useRef } from 'react';
import type { GameState, Move, BoardType } from '../../shared/types/game';
import type { MoveAnimationData } from './BoardView';
import { MoveHistory } from './MoveHistory';
import { GameFilters } from './ReplayPanel/GameFilters';
import { GameList } from './ReplayPanel/GameList';
import { PlaybackControls } from './ReplayPanel/PlaybackControls';
import { MoveInfo } from './ReplayPanel/MoveInfo';
import { useAuth } from '../contexts/AuthContext';
import {
  gameApi,
  type GameSummary,
  type GameHistoryResponse,
  type GameDetailsResponse,
} from '../services/api';
import { adaptHistoryToGameRecord } from '../services/ReplayService';
import { reconstructStateAtMove } from '../../shared/engine/replayHelpers';
import type { GameRecord } from '../../shared/types/gameRecord';
import type {
  ReplayGameQueryParams,
  ReplayGameMetadata,
  ReplayMoveRecord,
  PlaybackSpeed,
} from '../types/replay';

// ═════════════════════════════════════════════════════════════════════════════
// Types
// ═════════════════════════════════════════════════════════════════════════════

export interface ReplayPanelProps {
  /** Callback when replay state changes (pass null when exiting replay) */
  onStateChange: (state: GameState | null) => void;
  /** Callback when entering/exiting replay mode */
  onReplayModeChange: (inReplayMode: boolean) => void;
  /** Callback when forking from current position */
  onForkFromPosition: (state: GameState) => void;
  /** Callback for move animations */
  onAnimationChange?: (animation: MoveAnimationData | null) => void;
  /** Start collapsed */
  defaultCollapsed?: boolean;
  /** Optional className */
  className?: string;
}

// Number of games to show per page in the list
const DEFAULT_PAGE_SIZE = 10;

// Base delay between moves (ms) - divided by playback speed
const BASE_DELAY_MS = 1000;
// Minimum delay to keep animations readable at high speeds
const MIN_DELAY_MS = 200;

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Map a backend GameSummary into the generic ReplayGameMetadata shape expected by
 * the shared GameList component.
 *
 * Only a subset of fields are populated; others are left null/undefined.
 */
function mapSummaryToReplayMetadata(summary: GameSummary): ReplayGameMetadata {
  const numPlayers = summary.numPlayers ?? summary.playerCount;
  const terminationReason =
    (summary.outcome as string | undefined) ?? (summary.resultReason as string | undefined) ?? null;

  return {
    gameId: summary.id,
    boardType: summary.boardType as BoardType,
    numPlayers,
    winner: null, // Seat index is not cheaply available from the summary alone
    terminationReason,
    totalMoves: summary.moveCount,
    totalTurns: summary.moveCount,
    createdAt: summary.createdAt,
    completedAt: summary.endedAt ?? null,
    durationMs: null,
    source: summary.source ?? null,
    // v2 fields (time control, detailed players) are omitted for backend summaries
  };
}

/**
 * Apply simple client-side filters over the user's games. This keeps the backend
 * route cheap (status + pagination only) while giving a useful browsing UX.
 */
function applyFilters(games: GameSummary[], filters: ReplayGameQueryParams): GameSummary[] {
  return games.filter((game) => {
    if (filters.board_type && game.boardType !== filters.board_type) {
      return false;
    }

    const numPlayers = game.numPlayers ?? game.playerCount;
    if (filters.num_players && numPlayers !== filters.num_players) {
      return false;
    }

    const outcome: string | null =
      (game.outcome as string | undefined) ?? (game.resultReason as string | undefined) ?? null;

    if (filters.termination_reason && outcome !== filters.termination_reason) {
      return false;
    }

    if (filters.source && game.source !== filters.source) {
      return false;
    }

    return true;
  });
}

/**
 * Build lightweight ReplayMoveRecord entries for MoveInfo from the canonical
 * GameRecord + backend GameHistoryResponse.
 *
 * We rely on the canonical seat mapping from GameRecord.moves[*].player rather
 * than re-deriving it from history/details a second time.
 */
function buildReplayMoveRecords(
  record: GameRecord,
  history: GameHistoryResponse
): ReplayMoveRecord[] {
  const length = Math.min(record.moves.length, history.moves.length);
  const result: ReplayMoveRecord[] = [];

  for (let i = 0; i < length; i += 1) {
    const rec = record.moves[i] as any;
    const hist = history.moves[i];

    const metadata = (rec && typeof rec === 'object' && rec.metadata) || {};
    const thinkTimeMs: number | null =
      typeof metadata.thinkTimeMs === 'number'
        ? metadata.thinkTimeMs
        : typeof (hist as any).thinkTimeMs === 'number'
          ? (hist as any).thinkTimeMs
          : null;

    result.push({
      moveNumber: hist.moveNumber,
      turnNumber: hist.moveNumber,
      player: rec.player ?? 0,
      phase: (metadata.phase as string) ?? 'main',
      moveType: hist.moveType,
      move: hist.moveData,
      timestamp: hist.timestamp ?? null,
      thinkTimeMs,
    });
  }

  return result;
}

/**
 * Determine animation type from a canonical Move.
 */
function getAnimationTypeFromMove(move: Move): MoveAnimationData['type'] {
  switch (move.type) {
    case 'place_ring':
      return 'place';
    case 'overtaking_capture':
      return 'capture';
    case 'continue_capture_segment':
      return 'chain_capture';
    default:
      return 'move';
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main Component
// ═════════════════════════════════════════════════════════════════════════════

/**
 * ReplayPanel Component (backend multi-game replay browser)
 *
 * This panel lets the logged-in user:
 * - Browse their completed backend games (including imported self-play)
 * - Filter by board, players, outcome, and source
 * - Select a game and replay it using canonical GameRecord + reconstructStateAtMove
 *
 * The panel is read-only: it never submits moves, only drives the board via
 * onStateChange callbacks as the user scrubs through recorded moves.
 */
export function ReplayPanel({
  onStateChange,
  onReplayModeChange,
  onForkFromPosition,
  onAnimationChange,
  defaultCollapsed = true,
  className = '',
}: ReplayPanelProps) {
  const { user } = useAuth();

  // UI state
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  // Game browser state (backend /api/games/user/:id)
  const [filters, setFilters] = useState<ReplayGameQueryParams>({
    limit: DEFAULT_PAGE_SIZE,
    offset: 0,
  });
  const [userGames, setUserGames] = useState<GameSummary[]>([]);
  const [isLoadingGames, setIsLoadingGames] = useState(false);
  const [gamesError, setGamesError] = useState<string | null>(null);

  // Selected game for replay
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [selectedGameSource, setSelectedGameSource] = useState<string | undefined>(undefined);

  // Backend replay state (canonical record + local reconstruction)
  const [record, setRecord] = useState<GameRecord | null>(null);
  const [movesForDisplay, setMovesForDisplay] = useState<Move[]>([]);
  const [replayMoves, setReplayMoves] = useState<ReplayMoveRecord[]>([]);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [currentState, setCurrentState] = useState<GameState | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<PlaybackSpeed>(1);
  const [isLoadingReplay, setIsLoadingReplay] = useState(false);
  const [replayError, setReplayError] = useState<string | null>(null);

  // Timer for auto-play
  const playTimerRef = useRef<number | null>(null);

  // Derived state
  const inReplayMode = record !== null;
  const totalMoves = record?.moves.length ?? 0;
  const canStepBack = currentMoveIndex > 0;
  const canStepForward = currentMoveIndex < totalMoves;

  // ═══════════════════════════════════════════════════════════════════════════
  // Data loading
  // ═══════════════════════════════════════════════════════════════════════════

  // Load user's completed games when panel is expanded.
  useEffect(() => {
    if (isCollapsed) return;
    if (!user) return;
    if (userGames.length > 0 || isLoadingGames) return;

    const userId = user.id;
    let cancelled = false;

    async function loadUserGames() {
      try {
        setIsLoadingGames(true);
        setGamesError(null);

        const response = await gameApi.getUserGames(userId, {
          limit: 100,
          offset: 0,
          status: 'completed',
        });

        if (!cancelled) {
          setUserGames(response.games);
        }
      } catch (err) {
        if (!cancelled) {
          const message =
            err instanceof Error ? err.message : 'Failed to load your completed games';
          setGamesError(message);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingGames(false);
        }
      }
    }

    loadUserGames();

    return () => {
      cancelled = true;
    };
  }, [isCollapsed, user, userGames.length, isLoadingGames]);

  // Update parent with current GameState when it changes.
  useEffect(() => {
    onStateChange(currentState);
  }, [currentState, onStateChange]);

  // Notify parent when entering/exiting replay mode.
  useEffect(() => {
    onReplayModeChange(inReplayMode);
  }, [inReplayMode, onReplayModeChange]);

  // Clean up timer on unmount.
  useEffect(() => {
    return () => {
      if (playTimerRef.current !== null) {
        window.clearTimeout(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, []);

  // ═══════════════════════════════════════════════════════════════════════════
  // Replay helpers
  // ═══════════════════════════════════════════════════════════════════════════

  const clearPlaybackTimer = useCallback(() => {
    if (playTimerRef.current !== null) {
      window.clearTimeout(playTimerRef.current);
      playTimerRef.current = null;
    }
  }, []);

  const exitReplay = useCallback(() => {
    clearPlaybackTimer();

    setRecord(null);
    setMovesForDisplay([]);
    setReplayMoves([]);
    setCurrentMoveIndex(0);
    setCurrentState(null);
    setIsPlaying(false);
    setPlaybackSpeed(1);
    setIsLoadingReplay(false);
    setReplayError(null);

    onStateChange(null);
    onReplayModeChange(false);
    onAnimationChange?.(null);
  }, [clearPlaybackTimer, onStateChange, onReplayModeChange, onAnimationChange]);

  /**
   * Recompute GameState for a given move index using canonical replay helper.
   */
  const updateStateForIndex = useCallback((nextRecord: GameRecord, targetIndex: number) => {
    const clamped = Math.max(0, Math.min(targetIndex, nextRecord.moves.length));
    try {
      const nextState = reconstructStateAtMove(nextRecord, clamped);
      setCurrentState(nextState);
      setReplayError(null);
    } catch (err) {
      // Log for developers and surface a compact error to the user.

      console.error('ReplayPanel: failed to reconstruct state from GameRecord', err);
      setCurrentState(null);
      setReplayError('Failed to reconstruct game state for this replay.');
    }
  }, []);

  /**
   * Trigger a simple move animation based on the canonical Move[] used for
   * MoveHistory. This keeps replay animations consistent with sandbox board
   * animations without depending on the AI replay database.
   */
  const triggerAnimationForIndex = useCallback(
    (targetIndex: number) => {
      if (!onAnimationChange) return;

      if (targetIndex <= 0 || targetIndex > movesForDisplay.length) {
        onAnimationChange(null);
        return;
      }

      const move = movesForDisplay[targetIndex - 1] as Move;
      const from = move.from;
      const to = move.to;

      if (!to) {
        onAnimationChange(null);
        return;
      }

      const animation: MoveAnimationData = {
        type: getAnimationTypeFromMove(move),
        ...(from ? { from } : {}),
        to,
        playerNumber: move.player,
        id: `backend-replay-${targetIndex}`,
      };

      onAnimationChange(animation);
    },
    [movesForDisplay, onAnimationChange]
  );

  /**
   * Jump to a specific move index (0..totalMoves) and recompute state/animation.
   */
  const jumpToMove = useCallback(
    (target: number) => {
      if (!record) return;

      const clamped = Math.max(0, Math.min(target, record.moves.length));
      setCurrentMoveIndex(clamped);
      updateStateForIndex(record, clamped);
      triggerAnimationForIndex(clamped);
    },
    [record, updateStateForIndex, triggerAnimationForIndex]
  );

  const stepBack = useCallback(() => {
    if (!record) return;
    if (currentMoveIndex <= 0) return;
    jumpToMove(currentMoveIndex - 1);
  }, [record, currentMoveIndex, jumpToMove]);

  const stepForward = useCallback(() => {
    if (!record) return;
    if (currentMoveIndex >= record.moves.length) return;
    jumpToMove(currentMoveIndex + 1);
  }, [record, currentMoveIndex, jumpToMove]);

  const goToStart = useCallback(() => {
    if (!record) return;
    jumpToMove(0);
  }, [record, jumpToMove]);

  const goToEnd = useCallback(() => {
    if (!record) return;
    jumpToMove(record.moves.length);
  }, [record, jumpToMove]);

  const changeSpeed = useCallback((speed: PlaybackSpeed) => {
    setPlaybackSpeed(speed);
  }, []);

  const togglePlay = useCallback(() => {
    if (!record) return;

    setIsPlaying((prev) => {
      const next = !prev;

      if (!next) {
        // Turning off playback
        clearPlaybackTimer();
        return false;
      }

      // Starting playback from the beginning if at the end
      if (currentMoveIndex >= record.moves.length) {
        jumpToMove(0);
      }

      return true;
    });
  }, [record, currentMoveIndex, clearPlaybackTimer, jumpToMove]);

  // Auto-play timer
  useEffect(() => {
    if (!isPlaying || !record) {
      return;
    }

    if (currentMoveIndex >= record.moves.length) {
      setIsPlaying(false);
      return;
    }

    const delay = Math.max(MIN_DELAY_MS, BASE_DELAY_MS / playbackSpeed);

    const timer = window.setTimeout(() => {
      if (!record) return;

      const nextIndex = Math.min(currentMoveIndex + 1, record.moves.length);
      setCurrentMoveIndex(nextIndex);
      updateStateForIndex(record, nextIndex);
      triggerAnimationForIndex(nextIndex);

      if (nextIndex >= record.moves.length) {
        setIsPlaying(false);
      }
    }, delay);

    playTimerRef.current = timer;

    return () => {
      window.clearTimeout(timer);
      if (playTimerRef.current === timer) {
        playTimerRef.current = null;
      }
    };
  }, [
    isPlaying,
    playbackSpeed,
    currentMoveIndex,
    record,
    updateStateForIndex,
    triggerAnimationForIndex,
  ]);

  // Keyboard shortcuts (only when a replay is active)
  useEffect(() => {
    if (!inReplayMode) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;

      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          stepBack();
          break;
        case 'ArrowRight':
          e.preventDefault();
          stepForward();
          break;
        case ' ':
          e.preventDefault();
          togglePlay();
          break;
        case 'Home':
          e.preventDefault();
          goToStart();
          break;
        case 'End':
          e.preventDefault();
          goToEnd();
          break;
        case 'Escape':
          e.preventDefault();
          exitReplay();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [inReplayMode, stepBack, stepForward, togglePlay, goToStart, goToEnd, exitReplay]);

  /**
   * Load a specific game for replay by fetching history + details and projecting
   * them into the canonical GameRecord + Move[] that the shared replay engine
   * understands.
   */
  const loadReplayForGame = useCallback(
    async (gameId: string) => {
      clearPlaybackTimer();

      setIsLoadingReplay(true);
      setReplayError(null);

      try {
        const [history, details]: [GameHistoryResponse, GameDetailsResponse] = await Promise.all([
          gameApi.getGameHistory(gameId),
          gameApi.getGameDetails(gameId),
        ]);

        const { record: nextRecord, movesForDisplay: nextMoves } = adaptHistoryToGameRecord(
          history,
          details
        );
        const replayMoveRecords = buildReplayMoveRecords(nextRecord, history);

        setRecord(nextRecord);
        setMovesForDisplay(nextMoves);
        setReplayMoves(replayMoveRecords);

        const initialIndex = nextRecord.moves.length;
        setCurrentMoveIndex(initialIndex);
        updateStateForIndex(nextRecord, initialIndex);
        triggerAnimationForIndex(initialIndex);
        setIsPlaying(false);

        onReplayModeChange(true);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to load replay data for this game.';
        setReplayError(message);
        setRecord(null);
        setMovesForDisplay([]);
        setReplayMoves([]);
        setCurrentState(null);
        onStateChange(null);
        onReplayModeChange(false);
      } finally {
        setIsLoadingReplay(false);
      }
    },
    [
      clearPlaybackTimer,
      updateStateForIndex,
      triggerAnimationForIndex,
      onReplayModeChange,
      onStateChange,
    ]
  );

  // Select a game from the list
  const handleSelectGame = useCallback(
    (gameId: string) => {
      setSelectedGameId(gameId);
      const meta = userGames.find((g) => g.id === gameId);
      setSelectedGameSource(meta?.source);
      void loadReplayForGame(gameId);
    },
    [userGames, loadReplayForGame]
  );

  // Pagination handler for the local filtered list.
  const handlePageChange = useCallback((newOffset: number) => {
    setFilters((prev) => ({
      ...prev,
      offset: Math.max(0, newOffset),
    }));
  }, []);

  // Fork from current position into a new sandbox game.
  const handleFork = useCallback(() => {
    if (currentState) {
      onForkFromPosition(currentState);
    }
  }, [currentState, onForkFromPosition]);

  // ═══════════════════════════════════════════════════════════════════════════
  // Derived collections for GameList & MoveInfo
  // ═══════════════════════════════════════════════════════════════════════════

  const filteredGames = applyFilters(userGames, filters);
  const totalFiltered = filteredGames.length;
  const pageLimit = filters.limit ?? DEFAULT_PAGE_SIZE;
  const pageOffset = filters.offset ?? 0;
  const pageGames = filteredGames.slice(pageOffset, pageOffset + pageLimit);
  const hasMore = pageOffset + pageLimit < totalFiltered;

  const listGames: ReplayGameMetadata[] = pageGames.map(mapSummaryToReplayMetadata);

  const currentReplayMove: ReplayMoveRecord | null =
    currentMoveIndex > 0 && replayMoves.length >= currentMoveIndex
      ? replayMoves[currentMoveIndex - 1]
      : null;

  // Convert canonical Move list into the compact MoveHistory surface.
  const movesForHistory: Move[] = movesForDisplay;

  // ═══════════════════════════════════════════════════════════════════════════
  // Render
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div
      className={`border border-slate-700 rounded-2xl bg-slate-900/60 overflow-hidden ${className}`}
      data-testid="replay-panel"
    >
      {/* Header - always visible */}
      <button
        type="button"
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-slate-800/50 transition-colors"
      >
        <h2 className="font-semibold text-sm flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
              clipRule="evenodd"
            />
          </svg>
          Game Replay
          {inReplayMode && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-900/50 text-emerald-300">
              Active
            </span>
          )}
        </h2>
        <svg
          className={`w-4 h-4 transition-transform ${isCollapsed ? '' : 'rotate-180'}`}
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {/* Content - collapsible */}
      {!isCollapsed && (
        <div className="px-3 pb-3 space-y-3">
          {!inReplayMode ? (
            // Browse mode - user game list
            <>
              <p className="text-xs text-slate-400">
                Browse and replay your completed backend games, including imported self-play
                records.
              </p>

              {!user ? (
                <div className="text-xs text-slate-400 bg-slate-800/60 rounded px-2 py-2">
                  Log in to view and replay your completed games.
                </div>
              ) : (
                <>
                  {/* Filters */}
                  <GameFilters filters={filters} onFilterChange={setFilters} className="mt-1" />

                  {/* Game list */}
                  <GameList
                    games={listGames}
                    selectedGameId={selectedGameId}
                    onSelectGame={handleSelectGame}
                    isLoading={isLoadingGames}
                    error={gamesError}
                    total={totalFiltered}
                    offset={pageOffset}
                    limit={pageLimit}
                    hasMore={hasMore}
                    onPageChange={handlePageChange}
                  />
                </>
              )}
            </>
          ) : (
            // Replay mode - playback controls
            <>
              {/* Game info */}
              {record && (
                <div className="text-xs text-slate-400 bg-slate-800/50 rounded px-2 py-1.5">
                  <span className="font-medium text-slate-200">
                    {record.boardType} • {record.numPlayers}p
                  </span>
                  <span className="mx-2">|</span>
                  <span>{record.winner ? `P${record.winner} won` : 'draw'}</span>
                  <span className="mx-2">|</span>
                  <span>{selectedGameSource ?? 'online_game'}</span>
                </div>
              )}

              {/* Replay loading / error state */}
              {isLoadingReplay && (
                <div className="text-xs text-slate-400 bg-slate-900/60 rounded px-2 py-1.5">
                  Preparing replay…
                </div>
              )}

              {replayError && !isLoadingReplay && (
                <div className="text-xs text-red-400 bg-red-900/30 border border-red-700/50 rounded px-2 py-1.5">
                  Replay unavailable: {replayError}
                </div>
              )}

              {/* Playback controls */}
              {record && !isLoadingReplay && !replayError && (
                <>
                  <PlaybackControls
                    currentMove={currentMoveIndex}
                    totalMoves={totalMoves}
                    isPlaying={isPlaying}
                    playbackSpeed={playbackSpeed}
                    isLoading={false}
                    canStepForward={canStepForward}
                    canStepBackward={canStepBack}
                    onStepForward={stepForward}
                    onStepBackward={stepBack}
                    onJumpToStart={goToStart}
                    onJumpToEnd={goToEnd}
                    onJumpToMove={jumpToMove}
                    onTogglePlay={togglePlay}
                    onSetSpeed={changeSpeed}
                    className="mt-1"
                  />

                  {/* Current move info */}
                  <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700">
                    <MoveInfo move={currentReplayMove} moveNumber={currentMoveIndex} />
                  </div>

                  {/* Move history */}
                  {movesForHistory.length > 0 && (
                    <MoveHistory
                      moves={movesForHistory}
                      boardType={record.boardType as BoardType}
                      currentMoveIndex={currentMoveIndex > 0 ? currentMoveIndex - 1 : undefined}
                      onMoveClick={(index) => jumpToMove(index + 1)}
                      maxHeight="max-h-32"
                    />
                  )}

                  {/* Action buttons */}
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={handleFork}
                      disabled={!currentState}
                      className="flex-1 text-xs px-3 py-1.5 rounded bg-amber-600 hover:bg-amber-500 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors flex items-center justify-center gap-1"
                      title="Start playing from this position"
                    >
                      <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                        <path
                          fillRule="evenodd"
                          d="M7.707 3.293a1 1 0 010 1.414L5.414 7H11a7 7 0 017 7v2a1 1 0 11-2 0v-2a5 5 0 00-5-5H5.414l2.293 2.293a1 1 0 11-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                      Fork
                    </button>
                    <button
                      type="button"
                      onClick={exitReplay}
                      className="flex-1 text-xs px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 transition-colors"
                    >
                      Exit Replay
                    </button>
                  </div>

                  {/* Keyboard hints */}
                  <div className="text-[10px] text-slate-500 text-center">
                    <kbd className="px-1 py-0.5 bg-slate-800 rounded">Space</kbd> play/pause
                    <span className="mx-1">|</span>
                    <kbd className="px-1 py-0.5 bg-slate-800 rounded">←</kbd>{' '}
                    <kbd className="px-1 py-0.5 bg-slate-800 rounded">→</kbd> step
                    <span className="mx-1">|</span>
                    <kbd className="px-1 py-0.5 bg-slate-800 rounded">Esc</kbd> exit
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default ReplayPanel;
