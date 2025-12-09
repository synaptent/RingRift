import { useState } from 'react';
import type { GamePhase, Move, Player } from '../../shared/types/game';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import { PLAYER_COLORS, PHASE_INFO } from '../adapters/gameViewModels';
import { EvaluationGraph } from './EvaluationGraph';
import { MoveAnalysisPanel, type MoveAnalysis } from './MoveAnalysisPanel';

export interface SpectatorHUDProps {
  /** Current game phase */
  phase: GamePhase;
  /** All players in the game */
  players: Player[];
  /** Current player number (whose turn it is) */
  currentPlayerNumber: number;
  /** Current turn number */
  turnNumber: number;
  /** Current move number */
  moveNumber: number;
  /** Move history for annotations */
  moveHistory: Move[];
  /** Evaluation history for the graph */
  evaluationHistory: PositionEvaluationPayload['data'][];
  /** Currently selected move index for analysis */
  selectedMoveIndex?: number;
  /** Callback when a move is selected */
  onMoveSelect?: (moveIndex: number) => void;
  /** Number of spectators watching */
  spectatorCount?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get phase display info
 */
function getPhaseDisplay(phase: GamePhase): { label: string; colorClass: string; icon: string } {
  const info = PHASE_INFO[phase];
  if (info) {
    return {
      label: info.label,
      colorClass: info.colorClass,
      icon: info.icon,
    };
  }
  return { label: phase, colorClass: 'bg-slate-600', icon: '' };
}

/**
 * Get a brief annotation for a move
 */
function getMoveAnnotation(move: Move, playerNumber: number): string {
  const prefix = `P${playerNumber}`;
  switch (move.type) {
    case 'place_ring':
      return `${prefix} placed a ring`;
    case 'skip_placement':
    case 'no_placement_action':
      return `${prefix} skipped placement`;
    case 'move_ring':
    case 'move_stack':
    case 'build_stack':
    case 'no_movement_action':
      return `${prefix} moved a stack`;
    case 'overtaking_capture':
    case 'continue_capture_segment':
      return `${prefix} captured`;
    case 'skip_capture':
      return `${prefix} skipped capture`;
    case 'process_line':
    case 'choose_line_reward':
    case 'no_line_action':
      return `${prefix} claimed line bonus`;
    case 'process_territory_region':
    case 'eliminate_rings_from_stack':
    case 'skip_territory_processing':
    case 'no_territory_action':
      return `${prefix} processed territory`;
    case 'forced_elimination':
      return `${prefix} forced to eliminate`;
    case 'swap_sides':
      return `${prefix} swapped sides`;
    case 'line_formation':
      return `${prefix} formed a line`;
    case 'territory_claim':
      return `${prefix} claimed territory`;
    case 'recovery_slide':
      return `${prefix} performed recovery`;
    default:
      return `${prefix} made a move`;
  }
}
/**
 * Dedicated HUD for spectators watching a game.
 * Shows enhanced game state, move annotations, and integrated analysis.
 */
export function SpectatorHUD({
  phase,
  players,
  currentPlayerNumber,
  turnNumber,
  moveNumber,
  moveHistory,
  evaluationHistory,
  selectedMoveIndex,
  onMoveSelect,
  spectatorCount = 0,
  className = '',
}: SpectatorHUDProps) {
  const [showAnalysis, setShowAnalysis] = useState(true);

  const currentPlayer = players.find((p) => p.playerNumber === currentPlayerNumber);
  const currentPlayerName = currentPlayer?.username || `Player ${currentPlayerNumber}`;
  const currentPlayerColors = PLAYER_COLORS[currentPlayerNumber as keyof typeof PLAYER_COLORS] ?? {
    ring: 'bg-slate-300',
    hex: '#64748b',
  };

  const phaseDisplay = getPhaseDisplay(phase);
  const phaseInfo = PHASE_INFO[phase];

  // Get recent moves for annotation display
  const recentMoves = moveHistory.slice(-5);

  // Build analysis for selected move
  const hasSelectedMove = selectedMoveIndex !== undefined && !!moveHistory[selectedMoveIndex];

  const evaluationForSelected =
    hasSelectedMove && selectedMoveIndex !== undefined
      ? evaluationHistory.find((e) => e.moveNumber === selectedMoveIndex + 1)
      : undefined;

  const prevEvaluationForSelected =
    hasSelectedMove && selectedMoveIndex !== undefined && selectedMoveIndex > 0
      ? evaluationHistory.find((e) => e.moveNumber === selectedMoveIndex)
      : undefined;

  const selectedAnalysis: MoveAnalysis | null =
    hasSelectedMove && selectedMoveIndex !== undefined
      ? {
          move: moveHistory[selectedMoveIndex],
          moveNumber: selectedMoveIndex + 1,
          playerNumber: moveHistory[selectedMoveIndex].player,
          ...(evaluationForSelected ? { evaluation: evaluationForSelected } : {}),
          ...(prevEvaluationForSelected ? { prevEvaluation: prevEvaluationForSelected } : {}),
        }
      : null;

  return (
    <div className={`space-y-3 ${className}`} data-testid="spectator-hud">
      {/* Spectator Mode Header */}
      <div className="border border-purple-500/40 rounded-lg bg-purple-900/30 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-4 h-4 text-purple-300"
              aria-hidden="true"
            >
              <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
              <path
                fillRule="evenodd"
                d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-sm font-semibold text-purple-100">Spectator Mode</span>
          </div>
          {spectatorCount > 0 && (
            <span className="text-xs text-purple-200/70">
              {spectatorCount} {spectatorCount === 1 ? 'viewer' : 'viewers'}
            </span>
          )}
        </div>
      </div>

      {/* Game Status */}
      <div className="border border-slate-700 rounded-lg bg-slate-900/70 p-3">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-1 rounded text-xs font-medium ${phaseDisplay.colorClass} text-white`}
            >
              {phaseDisplay.icon} {phaseDisplay.label}
            </span>
          </div>
          <div className="text-xs text-slate-400">
            Turn {turnNumber} • Move #{moveNumber}
          </div>
        </div>

        {/* Current Player Indicator */}
        <div className="flex items-center gap-2 p-2 rounded bg-slate-800/60 border border-slate-700">
          <span className={`w-3 h-3 rounded-full ${currentPlayerColors.ring}`} aria-hidden="true" />
          <span className="text-sm font-semibold text-slate-100">{currentPlayerName}</span>
          <span className="text-xs text-slate-400">is playing</span>
        </div>

        {/* Phase Hint for Spectators */}
        {phaseInfo?.spectatorHint && (
          <div className="mt-2 text-[11px] text-slate-400 italic">{phaseInfo.spectatorHint}</div>
        )}
      </div>

      {/* Player Standings */}
      <div className="border border-slate-700 rounded-lg bg-slate-900/70 p-3">
        <h3 className="text-xs font-semibold text-slate-400 mb-2">Players</h3>
        <div className="space-y-2">
          {players.map((player) => {
            const isCurrentPlayer = player.playerNumber === currentPlayerNumber;
            const colors = PLAYER_COLORS[player.playerNumber as keyof typeof PLAYER_COLORS] ?? {
              ring: 'bg-slate-300',
              hex: '#64748b',
            };
            const ringsInHand = player.ringsInHand ?? 0;
            const eliminated = player.eliminatedRings ?? 0;
            const territory = player.territorySpaces ?? 0;

            return (
              <div
                key={player.id}
                className={`flex items-center justify-between p-2 rounded ${
                  isCurrentPlayer ? 'bg-blue-900/30 border border-blue-500/40' : 'bg-slate-800/40'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${colors.ring}`} aria-hidden="true" />
                  <span
                    className={`text-sm ${isCurrentPlayer ? 'font-semibold text-slate-100' : 'text-slate-300'}`}
                  >
                    {player.username || `Player ${player.playerNumber}`}
                  </span>
                  {player.type === 'ai' && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-slate-700 text-slate-400">
                      AI
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 text-[11px] text-slate-400">
                  <span title="Rings in hand">{ringsInHand} hand</span>
                  <span title="Rings eliminated" className="text-red-400/70">
                    {eliminated} cap
                  </span>
                  {territory > 0 && (
                    <span title="Territory spaces" className="text-emerald-400/70">
                      {territory} terr
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Analysis Toggle */}
      <button
        type="button"
        onClick={() => setShowAnalysis(!showAnalysis)}
        className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-slate-700 bg-slate-800/60 text-xs text-slate-300 hover:bg-slate-800 transition-colors"
      >
        <span className="font-medium">Analysis & Insights</span>
        <svg
          className={`w-4 h-4 transition-transform ${showAnalysis ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Analysis Section (Collapsible) */}
      {showAnalysis && (
        <div className="space-y-3">
          {/* Evaluation Graph */}
          {evaluationHistory.length > 0 && (
            <EvaluationGraph
              evaluationHistory={evaluationHistory}
              players={players}
              {...(selectedMoveIndex !== undefined ? { currentMoveIndex: selectedMoveIndex } : {})}
              {...(onMoveSelect ? { onMoveClick: onMoveSelect } : {})}
            />
          )}

          {/* Move Analysis Panel */}
          <MoveAnalysisPanel analysis={selectedAnalysis} players={players} />

          {/* Recent Moves with Annotations */}
          <div className="border border-slate-700 rounded-lg bg-slate-900/70 p-3">
            <h3 className="text-xs font-semibold text-slate-400 mb-2">Recent Moves</h3>
            <div className="space-y-1">
              {recentMoves.length === 0 ? (
                <div className="text-[11px] text-slate-500">No moves yet.</div>
              ) : (
                recentMoves.map((move, idx) => {
                  const actualIndex = moveHistory.length - recentMoves.length + idx;
                  const isSelected = selectedMoveIndex === actualIndex;
                  const colors = PLAYER_COLORS[move.player as keyof typeof PLAYER_COLORS] ?? {
                    ring: 'bg-slate-300',
                    hex: '#64748b',
                  };

                  return (
                    <button
                      key={actualIndex}
                      type="button"
                      onClick={() => onMoveSelect?.(actualIndex)}
                      className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left text-[11px] transition-colors ${
                        isSelected
                          ? 'bg-blue-900/40 border border-blue-500/40'
                          : 'hover:bg-slate-800/60'
                      }`}
                    >
                      <span className="text-slate-500 w-6">#{actualIndex + 1}</span>
                      <span className={`w-2 h-2 rounded-full ${colors.ring}`} aria-hidden="true" />
                      <span className="text-slate-300 flex-1">
                        {getMoveAnnotation(move, move.player)}
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
