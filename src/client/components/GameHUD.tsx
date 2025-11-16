import React from 'react';
import { GameState, PlayerChoice } from '../../shared/types/game';

interface GameHUDProps {
  gameState: GameState;
  isConnecting: boolean;
  pendingChoice: PlayerChoice | null;
  choiceTimeRemainingMs: number | null;
}

function formatMillis(ms: number | null | undefined): string {
  if (ms == null || ms < 0) return '--:--';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const mm = minutes.toString().padStart(2, '0');
  const ss = seconds.toString().padStart(2, '0');
  return `${mm}:${ss}`;
}

export const GameHUD: React.FC<GameHUDProps> = ({
  gameState,
  isConnecting,
  pendingChoice,
  choiceTimeRemainingMs
}) => {
  const players = [...gameState.players].sort((a, b) => a.playerNumber - b.playerNumber);

  // Derive per-player ring counts from GameState.board.stacks
  const ringsOnBoardByPlayer = new Map<number, number>();
  for (const stack of gameState.board.stacks.values()) {
    const count = ringsOnBoardByPlayer.get(stack.controllingPlayer) ?? 0;
    ringsOnBoardByPlayer.set(stack.controllingPlayer, count + stack.stackHeight);
  }

  const timeControl = gameState.timeControl;

  return (
    <div className="space-y-3 text-sm text-slate-100">
      <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
        <h2 className="font-semibold mb-2">Players & Clocks</h2>
        <ul className="space-y-2 text-xs">
          {players.map(player => {
            const isActive = player.playerNumber === gameState.currentPlayer;
            const ringsOnBoard = ringsOnBoardByPlayer.get(player.playerNumber) ?? 0;
            return (
              <li
                key={player.playerNumber}
                className={
                  'flex flex-col rounded px-2 py-1 border ' +
                  (isActive
                    ? 'border-emerald-400 bg-emerald-900/30'
                    : 'border-slate-700 bg-slate-900/40')
                }
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-slate-100">
                    P{player.playerNumber}: {player.username || `Player ${player.playerNumber}`} ({
                      player.type
                    })
                  </span>
                  {timeControl && (
                    <span className="ml-2 text-[11px] font-mono text-slate-200">
                      {formatMillis(player.timeRemaining)}
                    </span>
                  )}
                </div>
                <div className="mt-1 grid grid-cols-3 gap-1 text-[11px] text-slate-300">
                  <span>In hand: {player.ringsInHand}</span>
                  <span>On board: {ringsOnBoard}</span>
                  <span>Eliminated: {player.eliminatedRings}</span>
                  <span className="col-span-3">Territory: {player.territorySpaces}</span>
                  {player.type === 'ai' && player.aiProfile && (
                    <span className="col-span-3 text-[11px] text-slate-300">
                      AI difficulty {player.aiProfile.difficulty}
                      {player.aiProfile.mode ? ` • mode: ${player.aiProfile.mode}` : ''}
                      {player.aiProfile.aiType ? ` • type: ${player.aiProfile.aiType}` : ''}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
        <h2 className="font-semibold mb-2">Game Status</h2>
        <ul className="list-disc list-inside text-slate-200 space-y-1 text-xs">
          <li>Board type: {gameState.boardType}</li>
          <li>Game status: {gameState.gameStatus}</li>
          <li>Phase: {gameState.currentPhase}</li>
          <li>Current player: P{gameState.currentPlayer}</li>
          <li>WebSocket connection: {isConnecting ? 'connecting' : 'connected'}.</li>
          <li>
            {pendingChoice ? (
              <span>
                Pending choice:{' '}
                <span className="font-semibold">{pendingChoice.type}</span> for P
                {pendingChoice.playerNumber}
                {choiceTimeRemainingMs != null && (
                  <span>
                    {' '}
                    (time remaining: {Math.max(0, Math.ceil(choiceTimeRemainingMs / 1000))}s)
                  </span>
                )}
              </span>
            ) : (
              <span>No pending choice.</span>
            )}
          </li>
        </ul>
      </div>
    </div>
  );
};
