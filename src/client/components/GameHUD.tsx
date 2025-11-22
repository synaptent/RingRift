import React from 'react';
import { GameState, Player, GamePhase, BOARD_CONFIGS } from '../../shared/types/game';
import { ConnectionStatus } from '../contexts/GameContext';

interface GameHUDProps {
  gameState: GameState;
  currentPlayer: Player | undefined;
  instruction?: string;
  connectionStatus?: ConnectionStatus;
  isSpectator?: boolean;
  lastHeartbeatAt?: number | null;
}

export function GameHUD({
  gameState,
  currentPlayer,
  instruction,
  connectionStatus = 'connected',
  isSpectator = false,
  lastHeartbeatAt,
}: GameHUDProps) {
  if (!currentPlayer) return null;

  const spectatorCount = gameState.spectators.length;

  const getPhaseLabel = (phase: GamePhase) => {
    switch (phase) {
      case 'ring_placement':
        return 'Ring Placement';
      case 'movement':
        return 'Movement';
      case 'capture':
        return 'Capture';
      case 'chain_capture':
        return 'Chain Capture';
      case 'line_processing':
        return 'Line Processing';
      case 'territory_processing':
        return 'Territory Processing';
      default:
        return phase;
    }
  };

  const formatTime = (ms: number) => {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getPlayerColorClass = (playerNumber: number) => {
    switch (playerNumber) {
      case 1:
        return 'text-emerald-400 border-emerald-500';
      case 2:
        return 'text-sky-400 border-sky-500';
      case 3:
        return 'text-amber-400 border-amber-500';
      case 4:
        return 'text-fuchsia-400 border-fuchsia-500';
      default:
        return 'text-slate-400 border-slate-500';
    }
  };

  const getPlayerBgClass = (playerNumber: number) => {
    switch (playerNumber) {
      case 1:
        return 'bg-emerald-900/30';
      case 2:
        return 'bg-sky-900/30';
      case 3:
        return 'bg-amber-900/30';
      case 4:
        return 'bg-fuchsia-900/30';
      default:
        return 'bg-slate-800';
    }
  };

  const connectionLabel = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting…';
      case 'reconnecting':
        return 'Reconnecting…';
      case 'disconnected':
      default:
        return 'Disconnected';
    }
  };

  const HEARTBEAT_STALE_THRESHOLD_MS = 8000;
  const heartbeatAge = lastHeartbeatAt ? Date.now() - lastHeartbeatAt : null;
  const isHeartbeatStale =
    heartbeatAge !== null &&
    heartbeatAge > HEARTBEAT_STALE_THRESHOLD_MS &&
    connectionStatus === 'connected';

  const connectionColor =
    connectionStatus === 'connected' && !isHeartbeatStale
      ? 'text-emerald-300'
      : connectionStatus === 'reconnecting'
        ? 'text-amber-300'
        : 'text-rose-300';

  const boardConfig = BOARD_CONFIGS[gameState.boardType];

  return (
    <div className="w-full max-w-4xl mx-auto mb-4">
      <div className="flex items-center justify-between text-xs text-slate-300 mb-3">
        <div className={`font-semibold ${connectionColor}`}>
          Connection: {connectionLabel()}
          {isHeartbeatStale && (
            <span className="ml-1 text-[11px] text-amber-200">(awaiting update…)</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {spectatorCount > 0 && (
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="w-3 h-3"
              >
                <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                <path
                  fillRule="evenodd"
                  d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                  clipRule="evenodd"
                />
              </svg>
              {spectatorCount}
            </span>
          )}
          {isSpectator && (
            <span className="px-2 py-0.5 rounded-full bg-purple-900/40 border border-purple-500/40 text-purple-100 uppercase tracking-wide font-semibold">
              Spectator
            </span>
          )}
        </div>
      </div>
      {/* Current Turn Banner */}
      <div
        className={`
        flex items-center justify-between px-6 py-4 rounded-xl border-2 shadow-lg mb-4 transition-all duration-300
        ${getPlayerBgClass(currentPlayer.playerNumber)}
        ${getPlayerColorClass(currentPlayer.playerNumber)}
        ${isSpectator ? 'opacity-90' : 'opacity-100'}
      `}
      >
        <div className="flex flex-col">
          <span className="text-xs font-bold opacity-70 uppercase tracking-widest mb-1">
            Current Turn
          </span>
          <div className="flex items-center gap-3">
            <div
              className={`w-3 h-3 rounded-full ${getPlayerColorClass(currentPlayer.playerNumber).replace('text-', 'bg-').replace('border-', '')} animate-pulse`}
            />
            <span className="text-3xl font-black text-white tracking-tight">
              {currentPlayer.username || `Player ${currentPlayer.playerNumber}`}
            </span>
          </div>
        </div>

        <div className="flex flex-col items-end">
          <span className="text-xs font-bold opacity-70 uppercase tracking-widest mb-1">Phase</span>
          <span className="text-xl font-bold text-white">
            {getPhaseLabel(gameState.currentPhase)}
          </span>
        </div>
      </div>

      {/* Instruction Banner */}
      {instruction && (
        <div className="mb-4 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-center">
          <span className="text-slate-200 font-medium">{instruction}</span>
        </div>
      )}

      {/* Compact per-player summary row */}
      <div className="flex flex-wrap gap-2 mb-3 text-[11px] text-slate-200">
        {gameState.players.map((p) => {
          const ringsPerPlayer = boardConfig.ringsPerPlayer;
          const ringsOnBoard = Math.max(0, ringsPerPlayer - p.ringsInHand - p.eliminatedRings);
          const ringsInPlay = p.ringsInHand + ringsOnBoard;
          const isCurrent = p.playerNumber === gameState.currentPlayer;

          return (
            <div
              key={`summary-${p.playerNumber}`}
              className={`px-2 py-1 rounded-full border bg-slate-900/60 ${
                isCurrent
                  ? 'border-emerald-400 text-emerald-200'
                  : 'border-slate-600 text-slate-200'
              }`}
            >
              <span className="font-semibold mr-1">P{p.playerNumber}</span>
              <span className="mr-1">
                R {ringsInPlay}/{ringsPerPlayer}
              </span>
              <span>• T {p.territorySpaces}</span>
            </div>
          );
        })}
      </div>

      {/* Player Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {gameState.players.map((p) => {
          const isCurrent = p.playerNumber === gameState.currentPlayer;
          const colorClass = getPlayerColorClass(p.playerNumber);
          const ringsPerPlayer = boardConfig.ringsPerPlayer;
          const ringsOnBoard = Math.max(0, ringsPerPlayer - p.ringsInHand - p.eliminatedRings);

          return (
            <div
              key={p.playerNumber}
              className={`
                p-3 rounded-lg border transition-all
                ${isCurrent ? 'bg-slate-700 border-slate-500 shadow-md scale-105 z-10' : 'bg-slate-800 border-slate-700 opacity-80'}
              `}
            >
              <div className={`font-bold mb-1 ${colorClass.split(' ')[0]}`}>
                {p.username || `P${p.playerNumber}`}
                {p.type === 'ai' && <span className="ml-1 text-xs opacity-70">(AI)</span>}
              </div>

              <div className="flex justify-between text-sm text-slate-300">
                <span>Time:</span>
                <span
                  className={`font-mono font-bold ${isCurrent ? 'text-yellow-400' : 'text-slate-400'}`}
                >
                  {formatTime(p.timeRemaining)}
                </span>
              </div>

              <div className="flex justify-between text-sm text-slate-300">
                <span>Hand:</span>
                <span className="font-mono font-bold text-white">{p.ringsInHand}</span>
              </div>

              <div className="flex justify-between text-sm text-slate-300">
                <span>On board:</span>
                <span className="font-mono font-bold text-white">{ringsOnBoard}</span>
              </div>

              <div className="flex justify-between text-sm text-slate-300">
                <span>Eliminated:</span>
                <span className="font-mono font-bold text-red-400">{p.eliminatedRings}</span>
              </div>

              <div className="flex justify-between text-sm text-slate-300">
                <span>Territory:</span>
                <span className="font-mono font-bold text-blue-400">{p.territorySpaces}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
