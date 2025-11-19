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
      setUsers(response.users);
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
        <h1 className="text-3xl font-bold text-white mb-2">Leaderboard</h1>
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
                const winRate =
                  user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 100) : 0;

                return (
                  <tr key={user.id} className="hover:bg-slate-700/30 transition-colors">
                    <td className="px-6 py-4 text-center font-mono text-slate-500">{index + 1}</td>
                    <td className="px-6 py-4 font-medium text-white">{user.username}</td>
                    <td className="px-6 py-4 text-right font-mono text-emerald-400 font-bold">
                      {user.rating}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">{winRate}%</td>
                    <td className="px-6 py-4 text-right text-slate-300">{user.gamesPlayed}</td>
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
