import type { Move, Player } from '../../shared/types/game';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import { PLAYER_COLORS } from '../adapters/gameViewModels';

export interface MoveAnalysis {
  move: Move;
  moveNumber: number;
  playerNumber: number;
  evaluation?: PositionEvaluationPayload['data'];
  prevEvaluation?: PositionEvaluationPayload['data'];
  thinkTimeMs?: number;
  engineDepth?: number;
}

export interface MoveAnalysisPanelProps {
  analysis: MoveAnalysis | null;
  players: Player[];
  className?: string;
}

/**
 * Get a human-readable move type label
 */
function getMoveTypeLabel(move: Move): string {
  switch (move.type) {
    case 'place_ring':
      return 'Placement';
    case 'skip_placement':
    case 'no_placement_action':
      return 'Placement (no action)';
    case 'move_stack':
    case 'move_ring':
      return 'Movement';
    case 'overtaking_capture':
    case 'continue_capture_segment':
      return 'Capture';
    case 'skip_capture':
      return 'Skip Capture';
    case 'process_line':
    case 'choose_line_reward':
    case 'no_line_action':
      return 'Line Processing';
    case 'process_territory_region':
    case 'skip_territory_processing':
    case 'no_territory_action':
      return 'Territory';
    case 'eliminate_rings_from_stack':
      return 'Eliminate Rings';
    case 'forced_elimination':
      return 'Forced Elim';
    case 'swap_sides':
      return 'Swap Sides';
    case 'recovery_slide':
      return 'Recovery';
    default:
      return move.type.replace(/_/g, ' ');
  }
}

/**
 * Calculate evaluation delta between moves
 */
function calculateEvalDelta(
  current: PositionEvaluationPayload['data'] | undefined,
  previous: PositionEvaluationPayload['data'] | undefined,
  playerNumber: number
): {
  delta: number;
  quality: 'excellent' | 'good' | 'neutral' | 'inaccuracy' | 'mistake' | 'blunder';
} | null {
  if (!current || !previous) return null;

  const currentEval = current.perPlayer[playerNumber]?.totalEval ?? 0;
  const prevEval = previous.perPlayer[playerNumber]?.totalEval ?? 0;
  const delta = currentEval - prevEval;

  // Classify move quality based on evaluation change
  // Positive delta = good for the player
  let quality: 'excellent' | 'good' | 'neutral' | 'inaccuracy' | 'mistake' | 'blunder';
  if (delta >= 3) {
    quality = 'excellent';
  } else if (delta >= 1) {
    quality = 'good';
  } else if (delta >= -1) {
    quality = 'neutral';
  } else if (delta >= -3) {
    quality = 'inaccuracy';
  } else if (delta >= -6) {
    quality = 'mistake';
  } else {
    quality = 'blunder';
  }

  return { delta, quality };
}

/**
 * Get styling for move quality badge
 */
function getQualityStyle(quality: string): { bg: string; text: string; label: string } {
  switch (quality) {
    case 'excellent':
      return { bg: 'bg-emerald-500/30', text: 'text-emerald-300', label: 'Excellent' };
    case 'good':
      return { bg: 'bg-green-500/30', text: 'text-green-300', label: 'Good' };
    case 'neutral':
      return { bg: 'bg-slate-500/30', text: 'text-slate-300', label: 'Book' };
    case 'inaccuracy':
      return { bg: 'bg-amber-500/30', text: 'text-amber-300', label: 'Inaccuracy' };
    case 'mistake':
      return { bg: 'bg-orange-500/30', text: 'text-orange-300', label: 'Mistake' };
    case 'blunder':
      return { bg: 'bg-red-500/30', text: 'text-red-300', label: 'Blunder' };
    default:
      return { bg: 'bg-slate-500/30', text: 'text-slate-300', label: 'Unknown' };
  }
}

/**
 * Panel showing detailed analysis of a single move.
 * Displays move quality, evaluation change, think time, and position context.
 */
export function MoveAnalysisPanel({ analysis, players, className = '' }: MoveAnalysisPanelProps) {
  if (!analysis) {
    return (
      <div
        className={`border border-slate-700 rounded bg-slate-900/70 p-3 ${className}`}
        data-testid="move-analysis-panel"
      >
        <h3 className="text-xs font-semibold text-slate-400 mb-2">Move Analysis</h3>
        <div className="text-[11px] text-slate-500">Select a move to see detailed analysis.</div>
      </div>
    );
  }

  const { move, moveNumber, playerNumber, evaluation, prevEvaluation, thinkTimeMs, engineDepth } =
    analysis;
  const player = players.find((p) => p.playerNumber === playerNumber);
  const playerName = player?.username || `Player ${playerNumber}`;
  const colors = PLAYER_COLORS[playerNumber as keyof typeof PLAYER_COLORS] ?? {
    ring: 'bg-slate-300',
    hex: '#64748b',
  };

  const evalResult = calculateEvalDelta(evaluation, prevEvaluation, playerNumber);
  const qualityStyle = evalResult ? getQualityStyle(evalResult.quality) : null;

  // Format think time
  const formatThinkTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
  };

  // Get current player evaluation
  const currentEval = evaluation?.perPlayer[playerNumber]?.totalEval;
  const territoryEval = evaluation?.perPlayer[playerNumber]?.territoryEval;
  const ringEval = evaluation?.perPlayer[playerNumber]?.ringEval;

  return (
    <div
      className={`border border-slate-700 rounded bg-slate-900/70 p-3 ${className}`}
      data-testid="move-analysis-panel"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${colors.ring}`} aria-hidden="true" />
          <span className="text-sm font-semibold text-slate-100">{playerName}</span>
          <span className="text-xs text-slate-400">Move #{moveNumber}</span>
        </div>
        {qualityStyle && (
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-semibold ${qualityStyle.bg} ${qualityStyle.text}`}
          >
            {qualityStyle.label}
          </span>
        )}
      </div>

      {/* Move Type */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-slate-400">Type:</span>
        <span className="text-xs text-slate-200 font-medium">{getMoveTypeLabel(move)}</span>
      </div>

      {/* Evaluation Details */}
      {evaluation && (
        <div className="space-y-2 mb-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Position Eval:</span>
            <span
              className={`font-mono ${currentEval && currentEval > 0 ? 'text-emerald-300' : currentEval && currentEval < 0 ? 'text-rose-300' : 'text-slate-200'}`}
            >
              {currentEval !== undefined
                ? `${currentEval > 0 ? '+' : ''}${currentEval.toFixed(1)}`
                : '—'}
            </span>
          </div>

          {evalResult && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Eval Change:</span>
              <span
                className={`font-mono ${evalResult.delta > 0 ? 'text-emerald-300' : evalResult.delta < 0 ? 'text-rose-300' : 'text-slate-200'}`}
              >
                {evalResult.delta > 0 ? '+' : ''}
                {evalResult.delta.toFixed(1)}
              </span>
            </div>
          )}

          {(territoryEval !== undefined || ringEval !== undefined) && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Breakdown:</span>
              <span className="text-slate-300 text-[11px]">
                T:{' '}
                {territoryEval !== undefined
                  ? `${territoryEval >= 0 ? '+' : ''}${territoryEval.toFixed(1)}`
                  : '—'}
                {' / '}
                R:{' '}
                {ringEval !== undefined ? `${ringEval >= 0 ? '+' : ''}${ringEval.toFixed(1)}` : '—'}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Engine Stats */}
      {(thinkTimeMs !== undefined || engineDepth !== undefined) && (
        <div className="pt-2 border-t border-slate-700/50 space-y-1">
          {thinkTimeMs !== undefined && (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-500">Think Time:</span>
              <span className="text-slate-400 font-mono">{formatThinkTime(thinkTimeMs)}</span>
            </div>
          )}
          {engineDepth !== undefined && (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-500">Search Depth:</span>
              <span className="text-slate-400 font-mono">{engineDepth}</span>
            </div>
          )}
        </div>
      )}

      {/* No evaluation data message */}
      {!evaluation && (
        <div className="text-[11px] text-slate-500">No AI evaluation available for this move.</div>
      )}
    </div>
  );
}
