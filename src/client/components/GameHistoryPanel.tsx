import { useState, useEffect } from 'react';
import { gameApi, type GameHistoryResponse, type GameHistoryMove } from '../services/api';
import { Badge } from './ui/Badge';
import { formatVictoryReason } from '../adapters/gameViewModels';
import { BoardView } from './BoardView';
import { MoveHistory } from './MoveHistory';
import { HistoryPlaybackPanel } from './HistoryPlaybackPanel';
import { reconstructStateAtMove } from '../../shared/engine/replayHelpers';
import { adaptHistoryToGameRecord } from '../services/ReplayService';
import type { GameRecord } from '../../shared/types/gameRecord';
import type { GameState, Move } from '../../shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface GameHistoryPanelProps {
  /** Game ID to fetch history for */
  gameId: string;
  /** Whether the panel is initially collapsed */
  defaultCollapsed?: boolean;
  /** Optional class name for additional styling */
  className?: string;
  /** Called when an error occurs while fetching history */
  onError?: (error: Error) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format a move type for display
 */
function formatMoveType(moveType: string): string {
  const typeMap: Record<string, string> = {
    place_ring: 'Place Ring',
    skip_placement: 'Skip Placement',
    move_ring: 'Move Ring',
    move_stack: 'Move Stack',
    build_stack: 'Build Stack',
    overtaking_capture: 'Capture',
    continue_capture_segment: 'Continue Capture',
    process_line: 'Process Line',
    choose_line_reward: 'Line Reward',
    process_territory_region: 'Territory',
    eliminate_rings_from_stack: 'Eliminate Rings',
    line_formation: 'Line Formation',
    territory_claim: 'Territory Claim',
  };
  return typeMap[moveType] || moveType.replace(/_/g, ' ');
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Extract position description from move data
 */
function getPositionDescription(moveData: Record<string, unknown>): string {
  const parts: string[] = [];

  if (moveData.from && typeof moveData.from === 'object') {
    const from = moveData.from as { x: number; y: number; z?: number };
    parts.push(`from (${from.x},${from.y}${from.z !== undefined ? `,${from.z}` : ''})`);
  }

  if (moveData.to && typeof moveData.to === 'object') {
    const to = moveData.to as { x: number; y: number; z?: number };
    parts.push(`to (${to.x},${to.y}${to.z !== undefined ? `,${to.z}` : ''})`);
  }

  return parts.join(' → ');
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-Components
// ═══════════════════════════════════════════════════════════════════════════

interface MoveItemProps {
  move: GameHistoryMove;
}

function MoveItem({ move }: MoveItemProps) {
  const [expanded, setExpanded] = useState(false);

  const positionDesc = getPositionDescription(move.moveData);
  const hasDetails =
    Object.keys(move.moveData).length > 0 &&
    Object.keys(move.moveData).some((k) => !['id', 'type', 'player', 'from', 'to'].includes(k));

  const isAutoResolved = !!move.autoResolved;
  let autoResolvedText: string | null = null;
  let autoResolvedAriaLabel: string | null = null;

  if (isAutoResolved && move.autoResolved) {
    const reason = move.autoResolved.reason;

    let reasonDisplay: string;
    if (reason === 'timeout') reasonDisplay = 'timeout';
    else if (reason === 'disconnected') reasonDisplay = 'disconnect';
    else if (reason === 'fallback') reasonDisplay = 'fallback move';
    else reasonDisplay = reason;

    const baseLabel = `Auto-resolved (${reasonDisplay})`;
    autoResolvedText = baseLabel;

    const detailParts: string[] = [];
    if (move.autoResolved.choiceKind) {
      detailParts.push(String(move.autoResolved.choiceKind));
    }
    if (move.autoResolved.choiceType) {
      detailParts.push(String(move.autoResolved.choiceType));
    }

    if (detailParts.length > 0) {
      autoResolvedAriaLabel = `${baseLabel} – ${detailParts.join(' ')} decision`;
    } else {
      autoResolvedAriaLabel = baseLabel;
    }
  }

  return (
    <div className="border-b border-slate-700/50 last:border-b-0">
      <div
        className={`px-3 py-2 flex items-center gap-2 ${hasDetails ? 'cursor-pointer hover:bg-slate-800/50' : ''}`}
        onClick={() => hasDetails && setExpanded(!expanded)}
        role={hasDetails ? 'button' : undefined}
        tabIndex={hasDetails ? 0 : undefined}
        onKeyDown={(e) => {
          if (hasDetails && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            setExpanded(!expanded);
          }
        }}
      >
        {/* Move Number */}
        <span className="text-xs font-mono text-slate-500 w-8">#{move.moveNumber}</span>

        {/* Player */}
        <span className="text-xs font-semibold text-blue-400 w-16 truncate" title={move.playerName}>
          {move.playerName}
        </span>

        {/* Move Type + Auto-resolve badge (if present) */}
        <div className="text-xs text-slate-300 flex-1 flex items-center gap-2 min-w-0">
          <span className="truncate">{formatMoveType(move.moveType)}</span>
          {autoResolvedText && (
            <Badge
              variant="warning"
              className="shrink-0"
              data-testid="auto-resolved-badge"
              aria-label={autoResolvedAriaLabel ?? autoResolvedText}
            >
              {autoResolvedText}
            </Badge>
          )}
        </div>

        {/* Position (if available) */}
        {positionDesc && (
          <span className="text-xs text-slate-400 hidden sm:inline">{positionDesc}</span>
        )}

        {/* Timestamp */}
        <span className="text-[10px] text-slate-500">{formatTimestamp(move.timestamp)}</span>

        {/* Expand indicator */}
        {hasDetails && <span className="text-slate-500 text-xs">{expanded ? '▼' : '▶'}</span>}
      </div>

      {/* Expanded details */}
      {expanded && hasDetails && (
        <div className="px-3 py-2 bg-slate-800/30 text-xs">
          <pre className="text-slate-400 overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(move.moveData, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * GameHistoryPanel Component
 *
 * A collapsible panel that displays the complete move history for a game.
 * Fetches data from the API and displays moves in a scrollable list with
 * expandable details for each move.
 *
 * @example
 * ```tsx
 * <GameHistoryPanel
 *   gameId="abc123"
 *   defaultCollapsed={false}
 *   onError={(err) => console.error('Failed to load history:', err)}
 * />
 * ```
 */
export function GameHistoryPanel({
  gameId,
  defaultCollapsed = false,
  className = '',
  onError,
}: GameHistoryPanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<GameHistoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadVersion, setReloadVersion] = useState(0);

  // Backend replay state (single-game, backend-history based).
  const [replayOpen, setReplayOpen] = useState(false);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayRecord, setReplayRecord] = useState<GameRecord | null>(null);
  const [replayMoves, setReplayMoves] = useState<Move[]>([]);
  const [replayGameState, setReplayGameState] = useState<GameState | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [isViewingHistory, setIsViewingHistory] = useState(false);

  // Fetch history when panel is expanded or gameId changes
  useEffect(() => {
    if (collapsed || !gameId) return;

    let cancelled = false;

    async function fetchHistory() {
      setLoading(true);
      setError(null);

      try {
        const data = await gameApi.getGameHistory(gameId);
        if (!cancelled) {
          setHistory(data);
        }
      } catch (err) {
        if (!cancelled) {
          const errorMessage = err instanceof Error ? err.message : 'Failed to load history';
          setError(errorMessage);
          onError?.(err instanceof Error ? err : new Error(errorMessage));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchHistory();

    return () => {
      cancelled = true;
    };
  }, [gameId, collapsed, onError, reloadVersion]);

  // Recompute reconstructed GameState when replay record or index changes.
  useEffect(() => {
    if (!replayRecord || !replayOpen) {
      setReplayGameState(null);
      return;
    }

    const effectiveIndex = isViewingHistory ? currentMoveIndex : replayRecord.moves.length;

    try {
      const next = reconstructStateAtMove(replayRecord, effectiveIndex);
      setReplayGameState(next);
      setReplayError(null);
    } catch (err) {
      // Log for devs while surfacing a compact message in the UI.

      console.error('Failed to reconstruct replay state from backend history', err);
      setReplayGameState(null);
      setReplayError('Failed to reconstruct game state for replay.');
    }
  }, [replayRecord, replayOpen, currentMoveIndex, isViewingHistory]);

  const handleToggleReplay = async () => {
    if (!history || history.moves.length === 0) {
      return;
    }

    // Simple toggle when already initialized
    if (replayOpen) {
      setReplayOpen(false);
      return;
    }

    // If we already have a record, just reopen.
    if (replayRecord) {
      setReplayOpen(true);
      return;
    }

    setReplayLoading(true);
    setReplayError(null);

    try {
      const details = await gameApi.getGameDetails(gameId);
      const { record, movesForDisplay } = adaptHistoryToGameRecord(history, details);
      setReplayRecord(record);
      setReplayMoves(movesForDisplay);
      setCurrentMoveIndex(record.moves.length);
      setIsViewingHistory(false);
      setReplayOpen(true);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load replay metadata for this game';
      setReplayError(message);
      onError?.(err instanceof Error ? err : new Error(message));
    } finally {
      setReplayLoading(false);
    }
  };

  const handleMoveIndexChange = (index: number) => {
    if (!replayRecord) return;
    const clamped = Math.max(0, Math.min(index, replayRecord.moves.length));
    setCurrentMoveIndex(clamped);
  };

  const handleEnterHistoryView = () => {
    if (!replayRecord) return;
    setIsViewingHistory(true);
  };

  const handleExitHistoryView = () => {
    if (!replayRecord) return;
    setIsViewingHistory(false);
    setCurrentMoveIndex(replayRecord.moves.length);
  };

  const handleMoveClick = (index: number) => {
    if (!replayRecord) return;
    setIsViewingHistory(true);
    setCurrentMoveIndex(index + 1);
  };

  const hasReplaySnapshots = !!replayRecord && !!replayGameState;

  const activeMoveIndex =
    isViewingHistory && currentMoveIndex > 0 && replayRecord
      ? currentMoveIndex - 1
      : replayRecord
        ? replayRecord.moves.length - 1
        : undefined;

  return (
    <div
      className={`border border-slate-700 rounded-lg bg-slate-900/70 overflow-hidden ${className}`}
      data-testid="game-history-panel"
    >
      {/* Header */}
      <button
        className="w-full px-4 py-3 flex items-center justify-between bg-slate-800/50 hover:bg-slate-800/70 transition-colors"
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        aria-controls="game-history-content"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-200">Move History</span>
          {history && <span className="text-xs text-slate-400">({history.totalMoves} moves)</span>}
        </div>
        <span className="text-slate-400">{collapsed ? '▶' : '▼'}</span>
      </button>

      {/* Content */}
      {!collapsed && (
        <div
          id="game-history-content"
          className="max-h-80 overflow-y-auto"
          role="region"
          aria-label="Move history"
        >
          {/* Terminal result summary, when available */}
          {history?.result && !loading && !error && (
            <div className="px-4 py-2 border-b border-slate-700/50 bg-slate-900/60 text-xs text-slate-200 flex items-center justify-between">
              <span className="font-semibold">
                Result: {formatVictoryReason(history.result.reason)}
              </span>
              {history.result.winner !== undefined && history.result.winner !== null && (
                <span className="text-slate-400">Winner: P{history.result.winner}</span>
              )}
            </div>
          )}

          {/* Backend replay entry point for finished games */}
          {history && history.result && history.moves.length > 0 && !loading && !error && (
            <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-900/50 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-200">Replay this game</span>
                <button
                  type="button"
                  className="px-2 py-1 text-[11px] rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={handleToggleReplay}
                  disabled={replayLoading}
                  data-testid="open-replay-button"
                >
                  {replayOpen ? 'Hide replay' : 'Replay'}
                </button>
              </div>

              {replayOpen && (
                <div className="mt-2 space-y-3" data-testid="backend-replay-panel">
                  {replayLoading && (
                    <div className="text-[11px] text-slate-400">Preparing replay…</div>
                  )}

                  {replayError && !replayLoading && (
                    <div className="text-[11px] text-red-400">
                      Replay unavailable: {replayError}
                    </div>
                  )}

                  {!replayLoading && !replayError && replayRecord && (
                    <>
                      <HistoryPlaybackPanel
                        totalMoves={replayRecord.moves.length}
                        currentMoveIndex={currentMoveIndex}
                        isViewingHistory={isViewingHistory}
                        onMoveIndexChange={handleMoveIndexChange}
                        onExitHistoryView={handleExitHistoryView}
                        onEnterHistoryView={handleEnterHistoryView}
                        visible={true}
                        hasSnapshots={hasReplaySnapshots}
                      />

                      {replayGameState && (
                        <div className="flex flex-col md:flex-row gap-3">
                          <div className="flex-1 min-w-0 border-t border-slate-800 pt-3 md:border-t-0 md:border-r md:pr-3">
                            <BoardView
                              boardType={replayGameState.boardType}
                              board={replayGameState.board}
                              showCoordinateLabels={replayGameState.boardType === 'square8'}
                              showMovementGrid={false}
                              showLineOverlays={false}
                              showTerritoryRegionOverlays={false}
                            />
                          </div>
                          <div className="w-full md:w-56">
                            <MoveHistory
                              moves={replayMoves}
                              boardType={replayRecord.boardType}
                              currentMoveIndex={activeMoveIndex}
                              onMoveClick={handleMoveClick}
                              maxHeight="max-h-48"
                            />
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="px-4 py-8 text-center">
              <div className="animate-spin w-6 h-6 border-2 border-slate-600 border-t-blue-500 rounded-full mx-auto mb-2"></div>
              <span className="text-xs text-slate-400">Loading history...</span>
            </div>
          )}

          {/* Error state */}
          {error && !loading && (
            <div className="px-4 py-6 text-center">
              <div className="text-red-400 text-sm mb-2">⚠ {error}</div>
              <button
                className="text-xs text-blue-400 hover:text-blue-300 underline"
                onClick={() => {
                  // Clear previous history and error, then trigger a refetch.
                  setHistory(null);
                  setError(null);
                  setReloadVersion((v) => v + 1);
                }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Empty state */}
          {!loading && !error && history && history.moves.length === 0 && (
            <div className="px-4 py-6 text-center text-slate-400 text-sm">
              No moves recorded yet.
            </div>
          )}

          {/* Move list */}
          {!loading && !error && history && history.moves.length > 0 && (
            <div className="divide-y divide-slate-700/30">
              {history.moves.map((move) => (
                <MoveItem key={`${move.moveNumber}-${move.playerId}`} move={move} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default GameHistoryPanel;
