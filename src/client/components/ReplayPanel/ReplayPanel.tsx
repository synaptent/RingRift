/**
 * ReplayPanel - Main container for the game replay browser and playback controls.
 *
 * This is a collapsible panel that integrates into the sandbox sidebar.
 * It provides:
 * - Game database browser with filters
 * - Paginated game list
 * - Playback controls (step, play/pause, speed)
 * - Current move info display
 *
 * See: docs/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md
 */

import React, { useState, useCallback, useEffect } from 'react';
import { GameFilters } from './GameFilters';
import { GameList } from './GameList';
import { PlaybackControls } from './PlaybackControls';
import { MoveInfo } from './MoveInfo';
import { useGameList, useReplayServiceAvailable } from '../../hooks/useReplayService';
import { useReplayPlayback } from '../../hooks/useReplayPlayback';
import { useReplayAnimation } from '../../hooks/useReplayAnimation';
import type { ReplayGameQueryParams } from '../../types/replay';
import type { GameState } from '../../../shared/types/game';
import type { MoveAnimationData } from '../BoardView';

export interface ReplayPanelProps {
  /** Called when a game state should be displayed on the board */
  onStateChange?: (state: GameState | null) => void;
  /** Called when replay mode is entered/exited */
  onReplayModeChange?: (isInReplayMode: boolean) => void;
  /** Called when user wants to fork from current position */
  onForkFromPosition?: (state: GameState) => void;
  /** Called when an animation should be triggered */
  onAnimationChange?: (animation: MoveAnimationData | null) => void;
  /** Whether the panel starts collapsed */
  defaultCollapsed?: boolean;
  className?: string;
}

const DEFAULT_PAGE_SIZE = 10;

export function ReplayPanel({
  onStateChange,
  onReplayModeChange,
  onForkFromPosition,
  onAnimationChange,
  defaultCollapsed = true,
  className = '',
}: ReplayPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [filters, setFilters] = useState<ReplayGameQueryParams>({
    limit: DEFAULT_PAGE_SIZE,
    offset: 0,
  });

  // Check if replay service is available
  const { data: isAvailable, isLoading: isCheckingAvailability } = useReplayServiceAvailable();

  // Fetch game list
  const {
    data: gameListData,
    isLoading: isLoadingGames,
    error: gameListError,
  } = useGameList(filters, isAvailable === true && !isCollapsed);

  // Playback state
  const playback = useReplayPlayback();

  // Animation state for move transitions
  const { pendingAnimation } = useReplayAnimation({
    currentMoveNumber: playback.currentMoveNumber,
    moves: playback.moves,
    isPlaying: playback.isPlaying,
    enabled: playback.gameId !== null,
  });

  // Notify parent when state changes
  useEffect(() => {
    onStateChange?.(playback.currentState);
  }, [playback.currentState, onStateChange]);

  // Notify parent when entering/exiting replay mode
  useEffect(() => {
    onReplayModeChange?.(playback.gameId !== null);
  }, [playback.gameId, onReplayModeChange]);

  // Notify parent when animation changes
  useEffect(() => {
    onAnimationChange?.(pendingAnimation);
  }, [pendingAnimation, onAnimationChange]);

  // Handle game selection
  const handleSelectGame = useCallback(
    async (gameId: string) => {
      await playback.loadGame(gameId);
    },
    [playback]
  );

  // Handle page change
  const handlePageChange = useCallback((newOffset: number) => {
    setFilters((prev) => ({ ...prev, offset: newOffset }));
  }, []);

  // Handle close replay
  const handleCloseReplay = useCallback(() => {
    playback.unloadGame();
  }, [playback]);

  // Handle fork from position
  const handleFork = useCallback(() => {
    if (playback.currentState && onForkFromPosition) {
      onForkFromPosition(playback.currentState);
      playback.unloadGame();
    }
  }, [playback, onForkFromPosition]);

  // Keyboard shortcuts
  useEffect(() => {
    if (playback.gameId === null) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      switch (e.key) {
        case 'ArrowLeft':
        case 'h':
          e.preventDefault();
          playback.stepBackward();
          break;
        case 'ArrowRight':
        case 'l':
          e.preventDefault();
          playback.stepForward();
          break;
        case ' ':
          e.preventDefault();
          playback.togglePlay();
          break;
        case 'Home':
        case '0':
          e.preventDefault();
          playback.jumpToStart();
          break;
        case 'End':
        case '$':
          e.preventDefault();
          playback.jumpToEnd();
          break;
        case '[':
          e.preventDefault();
          if (playback.playbackSpeed === 1) playback.setSpeed(0.5);
          else if (playback.playbackSpeed === 2) playback.setSpeed(1);
          else if (playback.playbackSpeed === 4) playback.setSpeed(2);
          break;
        case ']':
          e.preventDefault();
          if (playback.playbackSpeed === 0.5) playback.setSpeed(1);
          else if (playback.playbackSpeed === 1) playback.setSpeed(2);
          else if (playback.playbackSpeed === 2) playback.setSpeed(4);
          break;
        case 'f':
          e.preventDefault();
          handleFork();
          break;
        case 'Escape':
          e.preventDefault();
          handleCloseReplay();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [playback, handleFork, handleCloseReplay]);

  // Render collapsed state
  if (isCollapsed) {
    return (
      <div className={`border border-slate-700 rounded-2xl bg-slate-900/60 ${className}`}>
        <button
          type="button"
          onClick={() => setIsCollapsed(false)}
          className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 rounded-2xl transition"
        >
          <div className="flex items-center gap-2">
            <span className="text-lg">📂</span>
            <span className="font-semibold text-sm text-slate-100">Game Database</span>
          </div>
          <span className="text-slate-400 text-xs">▼ Expand</span>
        </button>
      </div>
    );
  }

  // Service unavailable
  if (isCheckingAvailability) {
    return (
      <div className={`p-4 border border-slate-700 rounded-2xl bg-slate-900/60 ${className}`}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-sm text-slate-100 flex items-center gap-2">
            <span>📂</span> Game Database
          </h2>
          <button
            type="button"
            onClick={() => setIsCollapsed(true)}
            className="text-xs text-slate-400 hover:text-slate-200"
          >
            ▲ Collapse
          </button>
        </div>
        <p className="text-xs text-slate-400">Checking replay service...</p>
      </div>
    );
  }

  if (isAvailable === false) {
    return (
      <div className={`p-4 border border-slate-700 rounded-2xl bg-slate-900/60 ${className}`}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-sm text-slate-100 flex items-center gap-2">
            <span>📂</span> Game Database
          </h2>
          <button
            type="button"
            onClick={() => setIsCollapsed(true)}
            className="text-xs text-slate-400 hover:text-slate-200"
          >
            ▲ Collapse
          </button>
        </div>
        <p className="text-xs text-amber-400">
          Replay service unavailable. Start the AI service to browse stored games.
        </p>
        <p className="text-[10px] text-slate-500 mt-1">
          Run:{' '}
          <code className="bg-slate-800 px-1 rounded">
            cd ai-service && uvicorn app.main:app --port 8001
          </code>
        </p>
      </div>
    );
  }

  // Replay mode (game loaded)
  if (playback.gameId !== null) {
    const currentMove = playback.getCurrentMove();

    return (
      <div className={`p-4 border border-emerald-700/50 rounded-2xl bg-slate-900/60 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-sm text-emerald-100 flex items-center gap-2">
            <span>▶</span> Replay Mode
          </h2>
          <button
            type="button"
            onClick={handleCloseReplay}
            className="text-xs text-slate-400 hover:text-red-400 transition"
            title="Close replay (Esc)"
          >
            ✕ Close
          </button>
        </div>

        {/* Game info */}
        {playback.metadata && (
          <div className="mb-3 text-[10px] text-slate-400 space-y-0.5">
            <div className="flex items-center gap-2">
              <span className="font-mono">{playback.metadata.gameId.slice(0, 12)}...</span>
              <span className="px-1.5 py-0.5 rounded bg-slate-700">
                {playback.metadata.boardType}
              </span>
              <span>{playback.metadata.numPlayers}P</span>
            </div>
            {playback.metadata.winner !== null && (
              <div className="text-emerald-400">Winner: Player {playback.metadata.winner}</div>
            )}
          </div>
        )}

        {/* Error display */}
        {playback.error && (
          <div className="mb-3 p-2 rounded bg-red-900/40 border border-red-700/50 text-xs text-red-300">
            {playback.error}
          </div>
        )}

        {/* Playback controls */}
        <PlaybackControls
          currentMove={playback.currentMoveNumber}
          totalMoves={playback.totalMoves}
          isPlaying={playback.isPlaying}
          playbackSpeed={playback.playbackSpeed}
          isLoading={playback.isLoading}
          canStepForward={playback.canStepForward}
          canStepBackward={playback.canStepBackward}
          onStepForward={playback.stepForward}
          onStepBackward={playback.stepBackward}
          onJumpToStart={playback.jumpToStart}
          onJumpToEnd={playback.jumpToEnd}
          onJumpToMove={playback.jumpToMove}
          onTogglePlay={playback.togglePlay}
          onSetSpeed={playback.setSpeed}
          className="mb-3"
        />

        {/* Current move info */}
        <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700 mb-3">
          <MoveInfo move={currentMove} moveNumber={playback.currentMoveNumber} />
        </div>

        {/* Fork button */}
        {onForkFromPosition && playback.currentState && (
          <button
            type="button"
            onClick={handleFork}
            className="w-full px-3 py-2 rounded-lg border border-slate-600 text-xs font-medium text-slate-200 hover:border-emerald-500 hover:text-emerald-200 transition"
            title="Fork from here (f)"
          >
            Fork from this position
          </button>
        )}

        {/* Keyboard hints */}
        <div className="mt-3 text-[9px] text-slate-600 space-y-0.5">
          <p>← → Step • Space Play/Pause • [ ] Speed</p>
          <p>Home/End Jump • f Fork • Esc Close</p>
        </div>
      </div>
    );
  }

  // Browse mode (no game loaded)
  return (
    <div className={`p-4 border border-slate-700 rounded-2xl bg-slate-900/60 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-sm text-slate-100 flex items-center gap-2">
          <span>📂</span> Game Database
        </h2>
        <button
          type="button"
          onClick={() => setIsCollapsed(true)}
          className="text-xs text-slate-400 hover:text-slate-200"
        >
          ▲ Collapse
        </button>
      </div>

      {/* Filters */}
      <GameFilters filters={filters} onFilterChange={setFilters} className="mb-3" />

      {/* Game list */}
      <GameList
        games={gameListData?.games ?? []}
        selectedGameId={null}
        onSelectGame={handleSelectGame}
        isLoading={isLoadingGames}
        error={gameListError?.message ?? null}
        total={gameListData?.total ?? 0}
        offset={filters.offset ?? 0}
        limit={filters.limit ?? DEFAULT_PAGE_SIZE}
        hasMore={gameListData?.hasMore ?? false}
        onPageChange={handlePageChange}
      />
    </div>
  );
}
