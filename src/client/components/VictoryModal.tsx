import React from 'react';
import { GameResult, Player } from '../../shared/types/game';

interface VictoryModalProps {
  isOpen: boolean;
  gameResult: GameResult | null;
  players: Player[];
  onClose: () => void;
  onReturnToLobby: () => void;
}

export function VictoryModal({
  isOpen,
  gameResult,
  players,
  onClose,
  onReturnToLobby,
}: VictoryModalProps) {
  if (!isOpen || !gameResult) return null;

  const winner =
    gameResult.winner !== undefined
      ? players.find((p) => p.playerNumber === gameResult.winner)
      : null;

  const isDraw = gameResult.winner === undefined;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-slate-800 border border-slate-600 rounded-xl shadow-2xl p-8 max-w-md w-full text-center transform transition-all scale-100">
        <div className="mb-6">
          {isDraw ? (
            <div className="text-6xl mb-4">🤝</div>
          ) : (
            <div className="text-6xl mb-4">🏆</div>
          )}

          <h2 className="text-3xl font-bold text-white mb-2">
            {isDraw ? 'Game Drawn' : 'Victory!'}
          </h2>

          {winner && (
            <p className="text-xl text-emerald-400 font-semibold mb-1">
              {winner.username || `Player ${winner.playerNumber}`} wins!
            </p>
          )}

          <p className="text-slate-400 mt-4">{gameResult.reason}</p>
        </div>

        <div className="flex flex-col gap-3">
          <button
            onClick={onClose}
            className="w-full py-3 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
          >
            View Board
          </button>

          <button
            onClick={onReturnToLobby}
            className="w-full py-3 px-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-bold shadow-lg transition-colors"
          >
            Return to Lobby
          </button>
        </div>
      </div>
    </div>
  );
}
