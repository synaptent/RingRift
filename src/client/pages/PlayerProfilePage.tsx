import React, { useEffect, useState } from 'react';
import { useParams, Link, Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { userApi } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import { RatingHistoryChart } from '../components/RatingHistoryChart';

const BOARD_LABELS: Record<string, string> = {
  square8: 'Square 8x8',
  square19: 'Square 19x19',
  hex8: 'Hex Small',
  hexagonal: 'Hex Large',
};

interface PublicProfile {
  user: {
    id: string;
    username: string;
    rating: number;
    gamesPlayed: number;
    gamesWon: number;
    winRate: number;
    isProvisional: boolean;
    memberSince: string;
  };
  recentGames: Array<{
    id: string;
    boardType: string;
    winnerId: string | null;
    endedAt: string | null;
    maxPlayers: number;
    player1: { id: string; username: string } | null;
    player2: { id: string; username: string } | null;
    player3: { id: string; username: string } | null;
    player4: { id: string; username: string } | null;
  }>;
  ratingHistory: Array<{
    date: string;
    rating: number;
    change: number;
  }>;
}

export default function PlayerProfilePage() {
  const { userId } = useParams<{ userId: string }>();
  const { user: currentUser } = useAuth();
  const [profile, setProfile] = useState<PublicProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isOwnProfile = !!(userId && currentUser?.id === userId);

  useDocumentTitle(profile ? `${profile.user.username}'s Profile` : 'Player Profile');

  useEffect(() => {
    if (!userId || isOwnProfile) return;

    const fetchProfile = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const data = await userApi.getPublicProfile(userId);
        setProfile(data);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to load player profile';
        setError(message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchProfile();
  }, [userId, isOwnProfile]);

  // If viewing own profile, redirect to /profile
  if (isOwnProfile) {
    return <Navigate to="/profile" replace />;
  }

  if (!userId) {
    return <Navigate to="/leaderboard" replace />;
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="bg-red-900/30 border border-red-700 rounded-xl p-6 text-center">
          <p className="text-red-400">{error || 'Player not found'}</p>
          <Link
            to="/leaderboard"
            className="inline-block mt-4 text-sm text-sky-400 hover:text-sky-300 underline"
          >
            Back to Leaderboard
          </Link>
        </div>
      </div>
    );
  }

  const { user, recentGames, ratingHistory } = profile;

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl space-y-6">
      {/* Header */}
      <header className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-full bg-slate-700 flex items-center justify-center text-2xl font-bold text-emerald-400">
          {user.username.charAt(0).toUpperCase()}
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">{user.username}</h1>
          <p className="text-sm text-slate-400">
            Member since {new Date(user.memberSince).toLocaleDateString()}
          </p>
        </div>
      </header>

      {/* Stats cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Rating"
          value={user.rating.toString()}
          subtitle={user.isProvisional ? 'Provisional' : undefined}
          accent
        />
        <StatCard label="Games Played" value={user.gamesPlayed.toString()} />
        <StatCard label="Games Won" value={user.gamesWon.toString()} />
        <StatCard label="Win Rate" value={`${user.winRate}%`} />
      </div>

      {/* Rating history chart */}
      {ratingHistory.length > 1 && (
        <section className="bg-slate-800 rounded-xl border border-slate-700 p-5">
          <h2 className="text-lg font-semibold text-white mb-4">Rating History</h2>
          <RatingHistoryChart
            history={ratingHistory.map((e) => ({
              date: e.date,
              rating: e.rating,
              change: e.change,
              gameId: null,
            }))}
            currentRating={user.rating}
          />
        </section>
      )}

      {/* Recent games */}
      <section className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white">Recent Games</h2>
        </div>
        {recentGames.length === 0 ? (
          <div className="px-5 py-8 text-center text-slate-400">No completed games yet</div>
        ) : (
          <div className="divide-y divide-slate-700">
            {recentGames.map((game) => {
              const won = game.winnerId === userId;
              const drew = !game.winnerId;
              const players = [game.player1, game.player2, game.player3, game.player4].filter(
                Boolean
              );

              return (
                <div
                  key={game.id}
                  className="px-5 py-3 flex items-center justify-between hover:bg-slate-700/30 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                        won
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : drew
                            ? 'bg-slate-500/20 text-slate-400'
                            : 'bg-red-500/20 text-red-400'
                      }`}
                    >
                      {won ? 'W' : drew ? 'D' : 'L'}
                    </span>
                    <div>
                      <span className="text-sm text-white">
                        {BOARD_LABELS[game.boardType] || game.boardType}
                      </span>
                      <span className="text-xs text-slate-500 ml-2">
                        vs{' '}
                        {players
                          .filter((p) => p!.id !== userId)
                          .map((p) => p!.username)
                          .join(', ')}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs text-slate-500">
                    {game.endedAt ? new Date(game.endedAt).toLocaleDateString() : ''}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <div className="text-center">
        <Link to="/leaderboard" className="text-sm text-sky-400 hover:text-sky-300 underline">
          Back to Leaderboard
        </Link>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  subtitle,
  accent,
}: {
  label: string;
  value: string;
  subtitle?: string;
  accent?: boolean;
}) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 text-center">
      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">{label}</p>
      <p className={`text-xl font-bold ${accent ? 'text-emerald-400' : 'text-white'}`}>{value}</p>
      {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
    </div>
  );
}
