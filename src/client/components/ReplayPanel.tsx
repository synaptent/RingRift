import React, { useState, useEffect, useCallback, useRef } from 'react';
import type { GameState, Move, BoardType } from '../../shared/types/game';
import type { MoveAnimationData } from './BoardView';
import type { ReplayGameMetadata, ReplayPlaybackState, PlaybackSpeed } from '../types/replay';
import { getReplayService } from '../services/ReplayService';
import { MoveHistory } from './MoveHistory';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

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

// Available playback speeds
const PLAYBACK_SPEEDS: PlaybackSpeed[] = [0.5, 1, 2, 4];

// Base delay between moves (ms) - divided by playback speed
const BASE_DELAY_MS = 1000;

// ═══════════════════════════════════════════════════════════════════════════
// Sub-components
// ═══════════════════════════════════════════════════════════════════════════

interface PlaybackControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  onGoToStart: () => void;
  onGoToEnd: () => void;
  canStepBack: boolean;
  canStepForward: boolean;
}

function PlaybackControls({
  isPlaying,
  onTogglePlay,
  onStepBack,
  onStepForward,
  onGoToStart,
  onGoToEnd,
  canStepBack,
  canStepForward,
}: PlaybackControlsProps) {
  return (
    <div className="flex items-center justify-center gap-1">
      <button
        type="button"
        onClick={onGoToStart}
        disabled={!canStepBack}
        className="p-1.5 rounded hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Go to start (Home)"
        aria-label="Go to start"
      >
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path
            d="M4 5a1 1 0 011-1h1a1 1 0 011 1v10a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm5 0a1 1 0 011-1h4.586a1 1 0 01.707.293l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-.707.293H10a1 1 0 01-1-1V5z"
            transform="scale(-1,1) translate(-20,0)"
          />
        </svg>
      </button>

      <button
        type="button"
        onClick={onStepBack}
        disabled={!canStepBack}
        className="p-1.5 rounded hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Step back (Left arrow)"
        aria-label="Step back"
      >
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      <button
        type="button"
        onClick={onTogglePlay}
        disabled={!canStepForward && !isPlaying}
        className="p-2 rounded-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
              clipRule="evenodd"
            />
          </svg>
        )}
      </button>

      <button
        type="button"
        onClick={onStepForward}
        disabled={!canStepForward}
        className="p-1.5 rounded hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Step forward (Right arrow)"
        aria-label="Step forward"
      >
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      <button
        type="button"
        onClick={onGoToEnd}
        disabled={!canStepForward}
        className="p-1.5 rounded hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Go to end (End)"
        aria-label="Go to end"
      >
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path d="M4 5a1 1 0 011-1h1a1 1 0 011 1v10a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm5 0a1 1 0 011-1h4.586a1 1 0 01.707.293l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-.707.293H10a1 1 0 01-1-1V5z" />
        </svg>
      </button>
    </div>
  );
}

interface SpeedControlProps {
  currentSpeed: PlaybackSpeed;
  onSpeedChange: (speed: PlaybackSpeed) => void;
}

function SpeedControl({ currentSpeed, onSpeedChange }: SpeedControlProps) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-xs text-slate-400 mr-1">Speed:</span>
      <div className="flex rounded overflow-hidden border border-slate-600">
        {PLAYBACK_SPEEDS.map((speed) => (
          <button
            key={speed}
            type="button"
            onClick={() => onSpeedChange(speed)}
            className={`px-2 py-0.5 text-xs transition-colors ${
              currentSpeed === speed
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
            title={`${speed}x speed`}
            aria-pressed={currentSpeed === speed}
          >
            {speed}x
          </button>
        ))}
      </div>
    </div>
  );
}

interface ScrubberProps {
  currentMove: number;
  totalMoves: number;
  onSeek: (moveIndex: number) => void;
}

function Scrubber({ currentMove, totalMoves, onSeek }: ScrubberProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onSeek(parseInt(e.target.value, 10));
  };

  return (
    <div className="flex items-center gap-2 w-full">
      <span className="text-xs text-slate-400 min-w-[2.5rem] text-right">{currentMove}</span>
      <input
        type="range"
        min={0}
        max={totalMoves}
        value={currentMove}
        onChange={handleChange}
        className="flex-1 h-2 rounded-lg appearance-none cursor-pointer bg-slate-700
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-emerald-500
                   [&::-webkit-slider-thumb]:cursor-pointer
                   [&::-webkit-slider-thumb]:hover:bg-emerald-400
                   [&::-moz-range-thumb]:w-4
                   [&::-moz-range-thumb]:h-4
                   [&::-moz-range-thumb]:rounded-full
                   [&::-moz-range-thumb]:bg-emerald-500
                   [&::-moz-range-thumb]:cursor-pointer
                   [&::-moz-range-thumb]:border-0"
        aria-label="Move scrubber"
      />
      <span className="text-xs text-slate-400 min-w-[2.5rem]">{totalMoves}</span>
    </div>
  );
}

interface GameSelectorProps {
  games: ReplayGameMetadata[];
  isLoading: boolean;
  error: string | null;
  selectedGameId: string | null;
  onSelectGame: (gameId: string) => void;
  onRefresh: () => void;
}

function GameSelector({
  games,
  isLoading,
  error,
  selectedGameId,
  onSelectGame,
  onRefresh,
}: GameSelectorProps) {
  if (isLoading) {
    return <div className="text-center py-4 text-slate-400 text-xs">Loading games...</div>;
  }

  if (error) {
    return (
      <div className="text-center py-4">
        <p className="text-red-400 text-xs mb-2">{error}</p>
        <button
          type="button"
          onClick={onRefresh}
          className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600"
        >
          Retry
        </button>
      </div>
    );
  }

  if (games.length === 0) {
    return (
      <div className="text-center py-4 text-slate-400 text-xs">
        No recorded games found.
        <button
          type="button"
          onClick={onRefresh}
          className="block mx-auto mt-2 text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600"
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-1 max-h-32 overflow-y-auto">
      {games.slice(0, 10).map((game) => (
        <button
          key={game.gameId}
          type="button"
          onClick={() => onSelectGame(game.gameId)}
          className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors ${
            selectedGameId === game.gameId
              ? 'bg-emerald-900/50 border border-emerald-600'
              : 'bg-slate-800/50 hover:bg-slate-700/50 border border-transparent'
          }`}
        >
          <div className="flex justify-between items-center">
            <span className="font-medium truncate">
              {game.boardType} • {game.numPlayers}p
            </span>
            <span className="text-slate-400 ml-2">{game.totalMoves} moves</span>
          </div>
          <div className="text-slate-500 text-[10px] truncate">
            {game.source ?? 'unknown'} • {game.winner ? `P${game.winner} won` : 'draw'}
          </div>
        </button>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * ReplayPanel Component
 *
 * Self-contained panel for browsing and replaying recorded games.
 * Manages its own state and communicates back to parent via callbacks.
 *
 * Features:
 * - Collapsible panel UI
 * - Game browser with recent games
 * - Full playback controls (play/pause, step, scrubber)
 * - Speed control (0.5x to 4x)
 * - Click-to-jump move history
 * - Fork from position
 */
export function ReplayPanel({
  onStateChange,
  onReplayModeChange,
  onForkFromPosition,
  onAnimationChange,
  defaultCollapsed = true,
  className = '',
}: ReplayPanelProps) {
  // UI state
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [showGamePicker, setShowGamePicker] = useState(false);

  // Game browser state
  const [availableGames, setAvailableGames] = useState<ReplayGameMetadata[]>([]);
  const [isLoadingGames, setIsLoadingGames] = useState(false);
  const [gamesError, setGamesError] = useState<string | null>(null);

  // Playback state
  const [playbackState, setPlaybackState] = useState<ReplayPlaybackState>({
    gameId: null,
    metadata: null,
    currentMoveNumber: 0,
    totalMoves: 0,
    currentState: null,
    isPlaying: false,
    playbackSpeed: 1,
    isLoading: false,
    error: null,
    moves: [],
  });

  // Track when state reconstruction fails (diverges from backend)
  const [divergedAtMove, setDivergedAtMove] = useState<number | null>(null);

  // Refs
  const playTimerRef = useRef<number | null>(null);
  const replayServiceRef = useRef(getReplayService());

  // Derived state
  const inReplayMode = playbackState.gameId !== null;
  const canStepBack = playbackState.currentMoveNumber > 0;
  const canStepForward = playbackState.currentMoveNumber < playbackState.totalMoves;

  // Load available games
  const loadGames = useCallback(async () => {
    setIsLoadingGames(true);
    setGamesError(null);
    try {
      const response = await replayServiceRef.current.listGames({ limit: 20 });
      setAvailableGames(response.games);
    } catch (err) {
      setGamesError(err instanceof Error ? err.message : 'Failed to load games');
    } finally {
      setIsLoadingGames(false);
    }
  }, []);

  // Load games on mount when expanded
  useEffect(() => {
    if (!isCollapsed && availableGames.length === 0 && !isLoadingGames) {
      loadGames();
    }
  }, [isCollapsed, availableGames.length, isLoadingGames, loadGames]);

  // Fetch state at specific move
  const fetchStateAtMove = useCallback(async (gameId: string, moveNumber: number) => {
    try {
      const response = await replayServiceRef.current.getStateAtMove(gameId, moveNumber);
      return response.gameState;
    } catch (err) {
      console.error('Failed to fetch state at move', err);
      return null;
    }
  }, []);

  // Seek to specific move
  const seekToMove = useCallback(
    async (moveNumber: number) => {
      if (!playbackState.gameId) return;

      const clampedMove = Math.max(0, Math.min(moveNumber, playbackState.totalMoves));

      setPlaybackState((prev) => ({ ...prev, isLoading: true }));

      const state = await fetchStateAtMove(playbackState.gameId, clampedMove);

      setPlaybackState((prev) => ({
        ...prev,
        currentMoveNumber: clampedMove,
        currentState: state,
        isLoading: false,
      }));

      if (state) {
        onStateChange(state);

        // Generate animation if we have move data
        if (clampedMove > 0 && playbackState.moves.length > 0) {
          const moveRecord = playbackState.moves[clampedMove - 1];
          if (moveRecord && onAnimationChange) {
            // Extract animation data from move
            const moveData = moveRecord.move as {
              from?: { x: number; y: number };
              to?: { x: number; y: number };
            };
            if (moveData.from && moveData.to) {
              onAnimationChange({
                fromPosition: moveData.from,
                toPosition: moveData.to,
                playerNumber: moveRecord.player,
              });
            }
          }
        }

        // Clear divergence if we successfully got state
        if (divergedAtMove !== null && clampedMove < divergedAtMove) {
          setDivergedAtMove(null);
        }
      } else {
        // State reconstruction failed - track where it diverged
        if (divergedAtMove === null || clampedMove < divergedAtMove) {
          setDivergedAtMove(clampedMove);
        }
      }
    },
    [
      playbackState.gameId,
      playbackState.totalMoves,
      playbackState.moves,
      fetchStateAtMove,
      onStateChange,
      onAnimationChange,
      divergedAtMove,
    ]
  );

  // Load a game for replay
  const loadGame = useCallback(
    async (gameId: string) => {
      setPlaybackState((prev) => ({ ...prev, isLoading: true, error: null }));
      setDivergedAtMove(null); // Clear any previous divergence

      try {
        // Fetch game metadata
        const metadata = await replayServiceRef.current.getGame(gameId);

        // Fetch all moves
        const movesResponse = await replayServiceRef.current.getMoves(gameId, 0, undefined, 1000);

        // Fetch initial state
        const initialState = await fetchStateAtMove(gameId, 0);

        setPlaybackState({
          gameId,
          metadata,
          currentMoveNumber: 0,
          totalMoves: metadata.totalMoves,
          currentState: initialState,
          isPlaying: false,
          playbackSpeed: 1,
          isLoading: false,
          error: null,
          moves: movesResponse.moves,
        });

        onStateChange(initialState);
        onReplayModeChange(true);
        setShowGamePicker(false);
      } catch (err) {
        setPlaybackState((prev) => ({
          ...prev,
          isLoading: false,
          error: err instanceof Error ? err.message : 'Failed to load game',
        }));
      }
    },
    [fetchStateAtMove, onStateChange, onReplayModeChange]
  );

  // Exit replay mode
  const exitReplay = useCallback(() => {
    // Stop playback
    if (playTimerRef.current) {
      window.clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }

    setPlaybackState({
      gameId: null,
      metadata: null,
      currentMoveNumber: 0,
      totalMoves: 0,
      currentState: null,
      isPlaying: false,
      playbackSpeed: 1,
      isLoading: false,
      error: null,
      moves: [],
    });

    setDivergedAtMove(null);
    onStateChange(null);
    onReplayModeChange(false);
    onAnimationChange?.(null);
  }, [onStateChange, onReplayModeChange, onAnimationChange]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    setPlaybackState((prev) => {
      if (prev.isPlaying) {
        // Stop playback
        if (playTimerRef.current) {
          window.clearInterval(playTimerRef.current);
          playTimerRef.current = null;
        }
        return { ...prev, isPlaying: false };
      } else {
        // Start playback (only if we can step forward)
        if (prev.currentMoveNumber >= prev.totalMoves) {
          return prev;
        }
        return { ...prev, isPlaying: true };
      }
    });
  }, []);

  // Change playback speed
  const changeSpeed = useCallback((speed: PlaybackSpeed) => {
    setPlaybackState((prev) => ({ ...prev, playbackSpeed: speed }));
  }, []);

  // Step handlers
  const stepBack = useCallback(() => {
    if (canStepBack) {
      seekToMove(playbackState.currentMoveNumber - 1);
    }
  }, [canStepBack, playbackState.currentMoveNumber, seekToMove]);

  const stepForward = useCallback(() => {
    if (canStepForward) {
      seekToMove(playbackState.currentMoveNumber + 1);
    }
  }, [canStepForward, playbackState.currentMoveNumber, seekToMove]);

  const goToStart = useCallback(() => {
    seekToMove(0);
  }, [seekToMove]);

  const goToEnd = useCallback(() => {
    seekToMove(playbackState.totalMoves);
  }, [playbackState.totalMoves, seekToMove]);

  // Fork from current position
  const handleFork = useCallback(() => {
    if (playbackState.currentState) {
      onForkFromPosition(playbackState.currentState);
    }
  }, [playbackState.currentState, onForkFromPosition]);

  // Auto-play timer
  useEffect(() => {
    if (playbackState.isPlaying && inReplayMode) {
      const delay = BASE_DELAY_MS / playbackState.playbackSpeed;

      playTimerRef.current = window.setInterval(() => {
        setPlaybackState((prev) => {
          if (prev.currentMoveNumber >= prev.totalMoves) {
            // Reached end - stop playback
            if (playTimerRef.current) {
              window.clearInterval(playTimerRef.current);
              playTimerRef.current = null;
            }
            return { ...prev, isPlaying: false };
          }
          return prev;
        });

        // Advance to next move
        seekToMove(playbackState.currentMoveNumber + 1);
      }, delay);

      return () => {
        if (playTimerRef.current) {
          window.clearInterval(playTimerRef.current);
          playTimerRef.current = null;
        }
      };
    }
  }, [
    playbackState.isPlaying,
    playbackState.playbackSpeed,
    playbackState.currentMoveNumber,
    inReplayMode,
    seekToMove,
  ]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!inReplayMode) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
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

  // Convert ReplayMoveRecord to Move for MoveHistory
  const movesForHistory: Move[] = playbackState.moves.map((m) => ({
    type: m.moveType as Move['type'],
    playerNumber: m.player,
    ...(m.move as Record<string, unknown>),
  })) as Move[];

  // Render
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
            // Game selector mode
            <>
              <p className="text-xs text-slate-400">
                Browse and replay recorded games from AI training sessions.
              </p>

              {showGamePicker ? (
                <GameSelector
                  games={availableGames}
                  isLoading={isLoadingGames}
                  error={gamesError}
                  selectedGameId={null}
                  onSelectGame={loadGame}
                  onRefresh={loadGames}
                />
              ) : (
                <button
                  type="button"
                  onClick={() => {
                    setShowGamePicker(true);
                    if (availableGames.length === 0) {
                      loadGames();
                    }
                  }}
                  className="w-full px-3 py-2 rounded-lg bg-emerald-900/40 hover:bg-emerald-800/50 border border-emerald-700/50 text-sm text-emerald-200 transition-colors"
                >
                  Browse Recorded Games
                </button>
              )}
            </>
          ) : (
            // Replay mode - playback controls
            <>
              {/* Game info */}
              {playbackState.metadata && (
                <div className="text-xs text-slate-400 bg-slate-800/50 rounded px-2 py-1.5">
                  <span className="font-medium text-slate-200">
                    {playbackState.metadata.boardType} • {playbackState.metadata.numPlayers}p
                  </span>
                  <span className="mx-2">|</span>
                  <span>
                    {playbackState.metadata.winner
                      ? `P${playbackState.metadata.winner} won`
                      : 'draw'}
                  </span>
                  <span className="mx-2">|</span>
                  <span>{playbackState.metadata.source ?? 'unknown'}</span>
                </div>
              )}

              {/* State reconstruction failure warning */}
              {divergedAtMove !== null && (
                <div className="text-xs bg-amber-900/30 border border-amber-700/50 rounded px-2 py-1.5 text-amber-200">
                  <span className="font-medium">Replay diverged at move {divergedAtMove}</span>
                  <span className="text-amber-300/70">
                    {' '}
                    — unable to reconstruct further states. Board may be stale.
                  </span>
                </div>
              )}

              {/* Scrubber */}
              <Scrubber
                currentMove={playbackState.currentMoveNumber}
                totalMoves={playbackState.totalMoves}
                onSeek={seekToMove}
              />

              {/* Playback controls */}
              <PlaybackControls
                isPlaying={playbackState.isPlaying}
                onTogglePlay={togglePlay}
                onStepBack={stepBack}
                onStepForward={stepForward}
                onGoToStart={goToStart}
                onGoToEnd={goToEnd}
                canStepBack={canStepBack}
                canStepForward={canStepForward}
              />

              {/* Speed control */}
              <div className="flex justify-center">
                <SpeedControl
                  currentSpeed={playbackState.playbackSpeed}
                  onSpeedChange={changeSpeed}
                />
              </div>

              {/* Move history */}
              {movesForHistory.length > 0 && playbackState.metadata && (
                <MoveHistory
                  moves={movesForHistory}
                  boardType={playbackState.metadata.boardType as BoardType}
                  currentMoveIndex={
                    playbackState.currentMoveNumber > 0
                      ? playbackState.currentMoveNumber - 1
                      : undefined
                  }
                  onMoveClick={(index) => seekToMove(index + 1)}
                  maxHeight="max-h-32"
                />
              )}

              {/* Action buttons */}
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleFork}
                  disabled={!playbackState.currentState}
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
        </div>
      )}
    </div>
  );
}

export default ReplayPanel;
