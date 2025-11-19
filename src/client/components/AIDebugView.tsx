import React from 'react';
import { GameState } from '../../shared/types/game';

interface AIDebugViewProps {
  gameState: GameState;
  aiEvaluation?: {
    score: number;
    breakdown: Record<string, number>;
  };
  aiThinking?: boolean;
}

export const AIDebugView: React.FC<AIDebugViewProps> = ({
  gameState,
  aiEvaluation,
  aiThinking,
}) => {
  if (!aiEvaluation && !aiThinking) return null;

  return (
    <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3 text-sm text-slate-100">
      <h2 className="font-semibold flex items-center justify-between">
        <span>AI Analysis</span>
        {aiThinking && <span className="text-xs text-emerald-400 animate-pulse">Thinking...</span>}
      </h2>

      {aiEvaluation && (
        <div className="space-y-3">
          <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
            <span className="text-slate-400">Total Score</span>
            <span
              className={`font-mono font-bold ${
                aiEvaluation.score > 0
                  ? 'text-emerald-400'
                  : aiEvaluation.score < 0
                    ? 'text-red-400'
                    : 'text-slate-200'
              }`}
            >
              {aiEvaluation.score > 0 ? '+' : ''}
              {aiEvaluation.score.toFixed(2)}
            </span>
          </div>

          <div className="space-y-1">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Breakdown</p>
            {Object.entries(aiEvaluation.breakdown).map(
              ([key, value]) =>
                key !== 'total' && (
                  <div key={key} className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 capitalize">{key.replace(/_/g, ' ')}</span>
                    <span className="font-mono text-slate-400">
                      {value > 0 ? '+' : ''}
                      {value.toFixed(2)}
                    </span>
                  </div>
                )
            )}
          </div>
        </div>
      )}

      <div className="text-xs text-slate-500 mt-2">
        <p>
          AI Type:{' '}
          {gameState.players.find((p) => p.playerNumber === gameState.currentPlayer)?.aiProfile
            ?.aiType || 'Unknown'}
        </p>
        <p>
          Difficulty:{' '}
          {gameState.players.find((p) => p.playerNumber === gameState.currentPlayer)?.aiProfile
            ?.difficulty || '?'}
        </p>
      </div>
    </div>
  );
};
