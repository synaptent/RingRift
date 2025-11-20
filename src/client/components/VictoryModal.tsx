import React from 'react';
import { GameResult, Player, GameState } from '../../shared/types/game';

interface VictoryModalProps {
  isOpen: boolean;
  gameResult: GameResult | null;
  players: Player[];
  /**
   * Optional full GameState for computing additional summary information,
   * such as number of moves taken by the winner. When omitted, the modal
   * will fall back to basic winner stats only.
   */
  gameState?: GameState;
  onClose: () => void;
  onReturnToLobby: () => void;
}

function explainVictoryReason(reason: GameResult['reason']): string {
  switch (reason) {
    case 'ring_elimination':
      return 'Ring-elimination victory: one player has eliminated enough opposing rings to pass the victory threshold.';
    case 'territory_control':
      return 'Territory-control victory: under the territory and stalemate rules, this player finished with the strongest territory position on the board.';
    case 'last_player_standing':
      return 'Last-player-standing victory: all other players have no legal actions remaining.';
    case 'timeout':
      return 'Timeout victory: an opponent’s clock expired under the configured time control.';
    case 'resignation':
      return 'Resignation: an opponent voluntarily conceded the game.';
    case 'draw':
      return 'The game ended in a draw according to the stalemate and tiebreaker rules.';
    case 'abandonment':
      return 'Abandonment: the game was left in an unresolved state and adjudicated as abandoned.';
    case 'game_completed':
    default:
      return 'The game reached a completed state under the current rules.';
  }
}

export function VictoryModal({
  isOpen,
  gameResult,
  players,
  gameState,
  onClose,
  onReturnToLobby,
}: VictoryModalProps) {
  // Treat isOpen as an initial hint only; allow the user to dismiss the
  // modal locally via "Show Board" without requiring the parent to clear
  // the underlying victoryState.
  const [internalOpen, setInternalOpen] = React.useState(isOpen);

  if (!internalOpen || !gameResult) return null;

  const winner =
    gameResult.winner !== undefined
      ? players.find((p) => p.playerNumber === gameResult.winner)
      : null;

  const isDraw = gameResult.winner === undefined;

  let winnerMoveCount: number | undefined;
  if (winner && gameState) {
    // Prefer structured history entries; fall back to legacy moveHistory if needed.
    const historyMoves = gameState.history.filter(
      (entry) => entry.actor === winner.playerNumber
    ).length;
    const legacyMoves = gameState.moveHistory.filter(
      (m) => m.player === winner.playerNumber
    ).length;
    winnerMoveCount = historyMoves || legacyMoves || undefined;
  }

  const reasonText = explainVictoryReason(gameResult.reason);

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
            <>
              <p className="text-xl text-emerald-400 font-semibold mb-1">
                {winner.username || `Player ${winner.playerNumber}`} wins!
              </p>

              <div className="mt-4 text-sm text-slate-200 space-y-1">
                <p className="font-semibold">Winner stats</p>
                {typeof winnerMoveCount === 'number' && (
                  <p>
                    Moves played:{' '}
                    <span className="font-mono font-semibold text-white">
                      {winnerMoveCount}
                    </span>
                  </p>
                )}
                <p>
                  Rings in hand:{' '}
                  <span className="font-mono font-semibold text-white">
                    {winner.ringsInHand}
                  </span>
                </p>
                <p>
                  Eliminated rings:{' '}
                  <span className="font-mono font-semibold text-red-300">
                    {winner.eliminatedRings}
                  </span>
                </p>
                <p>
                  Territory spaces:{' '}
                  <span className="font-mono font-semibold text-sky-300">
                    {winner.territorySpaces}
                  </span>
                </p>
              </div>
            </>
          )}

          <div className="mt-4 text-sm text-slate-300">
            <p className="font-semibold">Victory condition</p>
            <p className="text-slate-400 mt-1">{reasonText}</p>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <button
            onClick={() => {
              setInternalOpen(false);
              onClose();
            }}
            className="w-full py-3 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
          >
            Show Board
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
