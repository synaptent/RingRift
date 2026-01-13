import React, { useEffect, useState } from 'react';
import { userApi } from '../services/api';
import { User } from '../../shared/types/user';
import LoadingSpinner from '../components/LoadingSpinner';

export default function LeaderboardPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  const fetchLeaderboard = async () => {
    try {
      setIsLoading(true);
      const response = await userApi.getLeaderboard({ limit: 50 });
      setUsers(response.users ?? []);
    } catch (err) {
      console.error('Failed to fetch leaderboard:', err);
      setError('Failed to load leaderboard data');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return <div className="container mx-auto px-4 py-8 text-center text-red-400">{error}</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
          <img src="/ringrift-icon.png" alt="RingRift" className="w-8 h-8" />
          Leaderboard
        </h1>
        <p className="text-slate-400">Top players ranked by rating</p>
      </header>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900/50 text-slate-400 uppercase tracking-wider font-medium">
              <tr>
                <th className="px-6 py-4 w-16 text-center">Rank</th>
                <th className="px-6 py-4">Player</th>
                <th className="px-6 py-4 text-right">Rating</th>
                <th className="px-6 py-4 text-right">Win Rate</th>
                <th className="px-6 py-4 text-right">Games</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {users.map((user, index) => {
                const gamesPlayed = user.gamesPlayed ?? 0;
                const gamesWon = user.gamesWon ?? 0;
                const winRate = gamesPlayed > 0 ? Math.round((gamesWon / gamesPlayed) * 100) : 0;
                const rank = index + 1;

                // Top 3 rank styling
                const rankBadge =
                  rank === 1 ? (
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-amber-500/20 text-amber-400 font-bold">
                      1
                    </span>
                  ) : rank === 2 ? (
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-slate-400/20 text-slate-300 font-bold">
                      2
                    </span>
                  ) : rank === 3 ? (
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-orange-700/20 text-orange-400 font-bold">
                      3
                    </span>
                  ) : (
                    <span className="font-mono text-slate-500">{rank}</span>
                  );

                return (
                  <tr
                    key={user.id}
                    className="hover:bg-slate-700/50 focus-within:bg-slate-700/50 transition-colors"
                    tabIndex={0}
                  >
                    <td className="px-6 py-4 text-center">{rankBadge}</td>
                    <td className="px-6 py-4 font-medium text-white">{user.username}</td>
                    <td className="px-6 py-4 text-right font-mono text-emerald-400 font-bold">
                      {user.rating ?? 1500}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">{winRate}%</td>
                    <td className="px-6 py-4 text-right text-slate-300">{gamesPlayed}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
