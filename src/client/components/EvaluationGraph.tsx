import { useMemo } from 'react';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import type { Player } from '../../shared/types/game';
import { PLAYER_COLORS } from '../adapters/gameViewModels';

export interface EvaluationGraphProps {
  evaluationHistory: PositionEvaluationPayload['data'][];
  players: Player[];
  currentMoveIndex?: number;
  onMoveClick?: (moveIndex: number) => void;
  className?: string;
  /** Height of the graph in pixels */
  height?: number;
}

/**
 * Visual graph showing evaluation history over the course of a game.
 * Displays per-player advantage lines that can be clicked to jump to moves.
 */
export function EvaluationGraph({
  evaluationHistory,
  players,
  currentMoveIndex,
  onMoveClick,
  className = '',
  height = 120,
}: EvaluationGraphProps) {
  const { points, minEval, maxEval, playerNumbers } = useMemo(() => {
    if (!evaluationHistory || evaluationHistory.length === 0) {
      return { points: [], minEval: -10, maxEval: 10, playerNumbers: [] };
    }

    // Get all unique player numbers
    const allPlayerNumbers = new Set<number>();
    for (const entry of evaluationHistory) {
      for (const pn of Object.keys(entry.perPlayer)) {
        allPlayerNumbers.add(Number.parseInt(pn, 10));
      }
    }
    const sortedPlayerNumbers = Array.from(allPlayerNumbers).sort((a, b) => a - b);

    // Calculate min/max for scaling
    let min = 0;
    let max = 0;
    for (const entry of evaluationHistory) {
      for (const ev of Object.values(entry.perPlayer)) {
        const total = ev.totalEval ?? 0;
        if (total < min) min = total;
        if (total > max) max = total;
      }
    }

    // Add padding and ensure symmetric around zero
    const absMax = Math.max(Math.abs(min), Math.abs(max), 5);
    const paddedMax = absMax * 1.2;

    return {
      points: evaluationHistory,
      minEval: -paddedMax,
      maxEval: paddedMax,
      playerNumbers: sortedPlayerNumbers,
    };
  }, [evaluationHistory]);

  if (points.length === 0) {
    return (
      <div
        className={`border border-slate-700 rounded bg-slate-900/70 p-3 ${className}`}
        data-testid="evaluation-graph"
      >
        <h3 className="text-xs font-semibold text-slate-400 mb-2">Evaluation Timeline</h3>
        <div className="text-[11px] text-slate-500">No evaluation data available yet.</div>
      </div>
    );
  }

  const width = 100; // percentage
  const graphHeight = height - 40; // Leave room for labels
  const padding = 4;

  // Scale helpers
  const scaleX = (moveNumber: number) => {
    const totalMoves = points[points.length - 1]?.moveNumber || 1;
    return padding + ((moveNumber - 1) / Math.max(totalMoves - 1, 1)) * (width - 2 * padding);
  };

  const scaleY = (evalValue: number) => {
    const range = maxEval - minEval;
    const normalized = (evalValue - minEval) / range;
    return graphHeight - normalized * graphHeight;
  };

  // Generate SVG path for each player
  const playerPaths = useMemo(() => {
    return playerNumbers
      .map((pn) => {
        const pathPoints = points
          .filter((p) => p.perPlayer[pn] !== undefined)
          .map((p) => {
            const x = scaleX(p.moveNumber);
            const y = scaleY(p.perPlayer[pn]?.totalEval ?? 0);
            return `${x},${y}`;
          });

        if (pathPoints.length < 2) return null;

        const colors = PLAYER_COLORS[pn as keyof typeof PLAYER_COLORS] ?? {
          ring: 'bg-slate-300',
          hex: '#64748b',
        };

        return {
          playerNumber: pn,
          path: `M ${pathPoints.join(' L ')}`,
          color: colors.hex,
        };
      })
      .filter(Boolean);
  }, [playerNumbers, points, maxEval, minEval, graphHeight]);

  // Get player name
  const getPlayerName = (pn: number) => {
    const player = players.find((p) => p.playerNumber === pn);
    return player?.username || `P${pn}`;
  };

  return (
    <div
      className={`border border-slate-700 rounded bg-slate-900/70 p-3 ${className}`}
      data-testid="evaluation-graph"
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-slate-400">Evaluation Timeline</h3>
        <div className="flex items-center gap-3">
          {playerNumbers.map((pn) => {
            const colors = PLAYER_COLORS[pn as keyof typeof PLAYER_COLORS] ?? {
              ring: 'bg-slate-300',
              hex: '#64748b',
            };
            return (
              <div key={pn} className="flex items-center gap-1 text-[10px] text-slate-400">
                <span className={`w-2 h-2 rounded-full ${colors.ring}`} aria-hidden="true" />
                <span>{getPlayerName(pn)}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="relative" style={{ height: graphHeight }}>
        {/* Zero line */}
        <div
          className="absolute left-0 right-0 border-t border-slate-600 border-dashed"
          style={{ top: `${scaleY(0)}px` }}
        />

        {/* SVG Graph */}
        <svg
          viewBox={`0 0 ${width} ${graphHeight}`}
          preserveAspectRatio="none"
          className="w-full h-full"
          style={{ overflow: 'visible' }}
        >
          {/* Player evaluation lines */}
          {playerPaths.map(
            (pp) =>
              pp && (
                <path
                  key={pp.playerNumber}
                  d={pp.path}
                  fill="none"
                  stroke={pp.color}
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                />
              )
          )}

          {/* Clickable points */}
          {onMoveClick &&
            points.map((p) => (
              <rect
                key={p.moveNumber}
                x={scaleX(p.moveNumber) - 2}
                y={0}
                width={4}
                height={graphHeight}
                fill="transparent"
                className="cursor-pointer hover:fill-slate-700/30"
                onClick={() => onMoveClick(p.moveNumber)}
              />
            ))}

          {/* Current move indicator */}
          {currentMoveIndex !== undefined && (
            <line
              x1={scaleX(currentMoveIndex + 1)}
              y1={0}
              x2={scaleX(currentMoveIndex + 1)}
              y2={graphHeight}
              stroke="#3b82f6"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>

        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 text-[9px] text-slate-500 -translate-x-1">
          +{maxEval.toFixed(0)}
        </div>
        <div
          className="absolute left-0 text-[9px] text-slate-500 -translate-x-1"
          style={{ top: `${scaleY(0)}px`, transform: 'translateY(-50%)' }}
        >
          0
        </div>
        <div className="absolute left-0 bottom-0 text-[9px] text-slate-500 -translate-x-1">
          {minEval.toFixed(0)}
        </div>
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between mt-1 text-[9px] text-slate-500">
        <span>Move 1</span>
        <span>Move {points[points.length - 1]?.moveNumber || 1}</span>
      </div>
    </div>
  );
}
