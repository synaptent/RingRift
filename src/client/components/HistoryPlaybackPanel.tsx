/**
 * HistoryPlaybackPanel - Playback controls for stepping through game move history
 *
 * This panel provides controls for scrubbing through the move history of a loaded
 * game (fixture/scenario). Unlike ReplayPanel which loads from the database API,
 * this works with locally-loaded games that have a move history.
 */

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';

export interface HistoryPlaybackPanelProps {
  /** Total number of moves in the history */
  totalMoves: number;
  /** Current move index being viewed (0 = initial state, totalMoves = final state) */
  currentMoveIndex: number;
  /** Whether currently viewing historical state (vs live game) */
  isViewingHistory: boolean;
  /** Callback to change the viewed move index */
  onMoveIndexChange: (index: number) => void;
  /** Callback to exit history view and return to live game */
  onExitHistoryView: () => void;
  /** Callback to enter history view mode */
  onEnterHistoryView: () => void;
  /** Whether the panel should be visible (show when there's history) */
  visible?: boolean;
  /**
   * Whether underlying snapshots exist for history playback. When false,
   * the panel renders in a disabled state with a hint instead of allowing
   * scrubbing that cannot change the board state.
   */
  hasSnapshots?: boolean;
}

type PlaybackSpeed = 0.5 | 1 | 2 | 5;
const PLAYBACK_SPEEDS: PlaybackSpeed[] = [0.5, 1, 2, 5];

export function HistoryPlaybackPanel({
  totalMoves,
  currentMoveIndex,
  isViewingHistory,
  onMoveIndexChange,
  onExitHistoryView,
  onEnterHistoryView,
  visible = true,
  hasSnapshots = true,
}: HistoryPlaybackPanelProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<PlaybackSpeed>(1);
  const playIntervalRef = useRef<number | null>(null);

  // Stop playback when we reach the end
  useEffect(() => {
    if (isPlaying && currentMoveIndex >= totalMoves) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentMoveIndex, totalMoves]);

  // Handle auto-play
  useEffect(() => {
    if (isPlaying && isViewingHistory) {
      const intervalMs = 1000 / playbackSpeed;
      playIntervalRef.current = window.setInterval(() => {
        onMoveIndexChange(currentMoveIndex + 1);
      }, intervalMs);
    }
  
    return () => {
      if (playIntervalRef.current !== null) {
        window.clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [isPlaying, isViewingHistory, currentMoveIndex, playbackSpeed, onMoveIndexChange]);

  // Stop playback when exiting history view
  useEffect(() => {
    if (!isViewingHistory) {
      setIsPlaying(false);
    }
  }, [isViewingHistory]);

  const handleStepBack = useCallback(() => {
    if (currentMoveIndex > 0) {
      if (!isViewingHistory) {
        onEnterHistoryView();
      }
      onMoveIndexChange(currentMoveIndex - 1);
    }
  }, [currentMoveIndex, isViewingHistory, onEnterHistoryView, onMoveIndexChange]);

  const handleStepForward = useCallback(() => {
    if (currentMoveIndex < totalMoves) {
      if (!isViewingHistory) {
        onEnterHistoryView();
      }
      onMoveIndexChange(currentMoveIndex + 1);
    }
  }, [currentMoveIndex, totalMoves, isViewingHistory, onEnterHistoryView, onMoveIndexChange]);

  const handlePlayPause = useCallback(() => {
    if (!isViewingHistory) {
      onEnterHistoryView();
      onMoveIndexChange(0); // Start from beginning
    }
    setIsPlaying((prev) => !prev);
  }, [isViewingHistory, onEnterHistoryView, onMoveIndexChange]);

  const handleJumpToStart = useCallback(() => {
    if (!isViewingHistory) {
      onEnterHistoryView();
    }
    onMoveIndexChange(0);
    setIsPlaying(false);
  }, [isViewingHistory, onEnterHistoryView, onMoveIndexChange]);

  const handleJumpToEnd = useCallback(() => {
    onMoveIndexChange(totalMoves);
    setIsPlaying(false);
    // If at end, can exit history view
    if (currentMoveIndex === totalMoves - 1) {
      onExitHistoryView();
    }
  }, [totalMoves, currentMoveIndex, onMoveIndexChange, onExitHistoryView]);

  const handleScrubberChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const newIndex = parseInt(e.target.value, 10);
      if (!isViewingHistory) {
        onEnterHistoryView();
      }
      onMoveIndexChange(newIndex);
    },
    [isViewingHistory, onEnterHistoryView, onMoveIndexChange]
  );

  // Keyboard shortcuts
  useEffect(() => {
    if (!visible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      const target = e.target as HTMLElement | null;
      if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA') return;

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          handleStepBack();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleStepForward();
          break;
        case ' ':
          e.preventDefault();
          handlePlayPause();
          break;
        case 'Home':
          e.preventDefault();
          handleJumpToStart();
          break;
        case 'End':
          e.preventDefault();
          handleJumpToEnd();
          break;
        case 'Escape':
          if (isViewingHistory) {
            e.preventDefault();
            onExitHistoryView();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    visible,
    isViewingHistory,
    handleStepBack,
    handleStepForward,
    handlePlayPause,
    handleJumpToStart,
    handleJumpToEnd,
    onExitHistoryView,
  ]);

  if (!visible || totalMoves === 0) {
    return null;
  }

  return (
    <div className="p-3 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-sm text-slate-200">History Playback</h2>
        {isViewingHistory && hasSnapshots && (
          <button
            type="button"
            onClick={onExitHistoryView}
            className="px-2 py-1 text-[10px] font-medium rounded bg-emerald-900/40 text-emerald-300 border border-emerald-700 hover:bg-emerald-800/40 transition"
          >
            Return to Live
          </button>
        )}
      </div>

      {/* Playback controls */}
      <div className="flex items-center justify-center gap-2">
        <button
          type="button"
          onClick={hasSnapshots ? handleJumpToStart : undefined}
          disabled={!hasSnapshots || (currentMoveIndex === 0 && isViewingHistory)}
          className="p-1.5 rounded hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
          title="Jump to start (Home)"
          aria-label="Jump to start"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4 5a1 1 0 011-1h2a1 1 0 011 1v10a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm8-1a1 1 0 00-1.447.894L8 10l2.553 5.106A1 1 0 0012 15V5a1 1 0 00-.447-.894z" />
          </svg>
        </button>

        <button
          type="button"
          onClick={hasSnapshots ? handleStepBack : undefined}
          disabled={!hasSnapshots || (currentMoveIndex === 0 && isViewingHistory)}
          className="p-1.5 rounded hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
          title="Step back (←)"
          aria-label="Step back"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" />
          </svg>
        </button>

        <button
          type="button"
          onClick={hasSnapshots ? handlePlayPause : undefined}
          disabled={!hasSnapshots || (currentMoveIndex >= totalMoves && isViewingHistory)}
          className="p-2 rounded-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition"
          title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
            </svg>
          )}
        </button>

        <button
          type="button"
          onClick={hasSnapshots ? handleStepForward : undefined}
          disabled={!hasSnapshots || currentMoveIndex >= totalMoves}
          className="p-1.5 rounded hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
          title="Step forward (→)"
          aria-label="Step forward"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" />
          </svg>
        </button>

        <button
          type="button"
          onClick={hasSnapshots ? handleJumpToEnd : undefined}
          disabled={!hasSnapshots || currentMoveIndex >= totalMoves}
          className="p-1.5 rounded hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
          title="Jump to end (End)"
          aria-label="Jump to end"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M16 5a1 1 0 00-1 1v10a1 1 0 002 0V6a1 1 0 00-1-1zm-8 1a1 1 0 011.447-.894L12 10l-2.553 5.106A1 1 0 018 15V6z" />
          </svg>
        </button>
      </div>

      {/* Scrubber */}
      <div className="space-y-1">
        <input
          type="range"
          min={0}
          max={totalMoves}
          value={isViewingHistory ? currentMoveIndex : totalMoves}
          onChange={hasSnapshots ? handleScrubberChange : undefined}
          className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:cursor-not-allowed"
          aria-label="Move scrubber"
          disabled={!hasSnapshots}
        />
        <div className="flex justify-between text-[10px] text-slate-400">
          <span>Start</span>
          <span>
            Move {isViewingHistory ? currentMoveIndex : totalMoves} / {totalMoves}
          </span>
          <span>End</span>
        </div>
      </div>

      {/* Speed control */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-400">Speed:</span>
        <div className="flex gap-1">
          {PLAYBACK_SPEEDS.map((speed) => (
            <button
              key={speed}
              type="button"
              onClick={() => setPlaybackSpeed(speed)}
              className={`px-2 py-0.5 rounded text-[10px] font-medium transition ${
                playbackSpeed === speed
                  ? 'bg-emerald-900/40 text-emerald-300 border border-emerald-700'
                  : 'bg-slate-800 text-slate-400 border border-slate-600 hover:border-slate-500'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {!hasSnapshots && (
        <p className="text-[10px] text-slate-500">
          History scrubbing is unavailable for this scenario (no internal snapshots were recorded).
        </p>
      )}

      {/* Keyboard hints */}
      <div className="text-[9px] text-slate-500 flex flex-wrap gap-x-3 gap-y-1">
        <span>
          <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">←</kbd>
          <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400 ml-0.5">→</kbd> Step
        </span>
        <span>
          <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">Space</kbd> Play/Pause
        </span>
        <span>
          <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">Esc</kbd> Exit
        </span>
      </div>
    </div>
  );
}

export default HistoryPlaybackPanel;
