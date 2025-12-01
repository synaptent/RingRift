import React, { useState, useEffect } from 'react';
import { gameApi, GameHistoryResponse, GameHistoryMove } from '../services/api';
import { Badge } from './ui/Badge';

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
  let autoResolvedLabel: string | null = null;

  if (isAutoResolved && move.autoResolved) {
    const reason = move.autoResolved.reason;

    let reasonDisplay: string;
    if (reason === 'timeout') reasonDisplay = 'timeout';
    else if (reason === 'disconnected') reasonDisplay = 'disconnect';
    else if (reason === 'fallback') reasonDisplay = 'fallback move';
    else reasonDisplay = reason;

    autoResolvedLabel = `Auto-resolved (${reasonDisplay})`;
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
          {autoResolvedLabel && (
            <Badge
              variant="warning"
              className="shrink-0"
              data-testid="auto-resolved-badge"
              aria-label={autoResolvedLabel}
            >
              {autoResolvedLabel}
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
  }, [gameId, collapsed, onError]);

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
                  setHistory(null);
                  setCollapsed(false);
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
