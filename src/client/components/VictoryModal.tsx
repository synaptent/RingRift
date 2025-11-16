import React from 'react';
import { GameResult, GameState } from '../../shared/types/game';

interface VictoryModalProps {
  isOpen: boolean;
  gameState: GameState | null;
  result: GameResult | null;
  onClose: () => void;
}

/**
 * Minimal victory modal shown when the backend emits a game_over event.
 *
 * This is intentionally simple but grounded in shared types so it can be
 * extended later (ratings, richer score breakdowns, rematch controls, etc.)
 * without changing the core wiring.
 */
export function VictoryModal({ isOpen, gameState, result, onClose }: VictoryModalProps) {
  if (!isOpen || !result) return null;

  const winnerNumber = result.winner;
  const winnerPlayer =
    winnerNumber !== undefined && gameState
      ? gameState.players.find(p => p.playerNumber === winnerNumber)
      : undefined;

  const reasonLabel = (() => {
    switch (result.reason) {
      case 'ring_elimination':
        return 'Ring elimination';
      case 'territory_control':
        return 'Territory control';
      case 'last_player_standing':
        return 'Last player standing';
      case 'timeout':
        return 'Time expired';
      case 'resignation':
        return 'Resignation';
      case 'draw':
        return 'Draw';
      case 'abandonment':
        return 'Abandonment';
      case 'game_completed':
      default:
        return 'Game completed';
    }
  })();

  const title = winnerPlayer
    ? `Winner: ${winnerPlayer.username || `Player ${winnerPlayer.playerNumber}`}`
    : result.reason === 'draw'
    ? 'Draw'
    : 'Game Over';

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70">
      <div className="max-w-md w-full mx-4 rounded-lg bg-slate-900 border border-slate-700 shadow-xl p-6 text-slate-100">
        <h2 className="text-2xl font-bold mb-2 text-center">{title}</h2>
        <p className="text-sm text-slate-300 mb-4 text-center">Reason: {reasonLabel}</p>

        {gameState && (
          <div className="mb-4 text-sm">
            <h3 className="font-semibold mb-1">Final scores (rings remaining + territory)</h3>
            <ul className="space-y-1">
              {gameState.players
                .slice()
                .sort((a, b) => a.playerNumber - b.playerNumber)
                .map(player => {
                  const pn = player.playerNumber;
                  const ringsRemaining = result.finalScore.ringsRemaining[pn] ?? 0;
                  const territory = result.finalScore.territorySpaces[pn] ?? 0;
                  return (
                    <li key={pn} className="flex justify-between">
                      <span>
                        P{pn}: {player.username || `Player ${pn}`}
                      </span>
                      <span>
                        {ringsRemaining} rings, {territory} territory
                      </span>
                    </li>
                  );
                })}
            </ul>
          </div>
        )}

        <div className="flex justify-center space-x-3 mt-2">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold"
          >
            Return to lobby
          </button>
        </div>
      </div>
    </div>
  );
}
