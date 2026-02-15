import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { userApi } from '../services/api';
import { User } from '../../shared/types/user';
import LoadingSpinner from '../components/LoadingSpinner';
import { LeaderboardSkeleton } from '../components/Skeleton';

interface SearchResult {
  id: string;
  username: string;
  rating: number;
  gamesPlayed: number;
  gamesWon: number;
}

export default function LeaderboardPage() {
  useDocumentTitle(
    'Leaderboard',
    'Top RingRift players ranked by Elo rating. See who leads on every board type.'
  );
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);

    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
    }

    if (query.length < 2) {
      setSearchResults(null);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    searchTimerRef.current = setTimeout(async () => {
      try {
        const result = await userApi.searchUsers(query, 10);
        setSearchResults(result.users);
      } catch {
        // Silently fail - leaderboard still visible
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);
  }, []);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, []);

  const displayUsers = searchResults ?? users;
  const showingSearch = searchResults !== null;

  if (isLoading) {
    return <LeaderboardSkeleton />;
  }

  if (error) {
    return <div className="container mx-auto px-4 py-8 text-center text-red-400">{error}</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
          <img src="/ringrift-icon.png" alt="RingRift" className="w-8 h-8" />
          Leaderboard
        </h1>
        <p className="text-slate-400">Top players ranked by rating</p>
      </header>

      {/* Search bar */}
      <div className="mb-4 relative">
        <div className="relative">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
            />
          </svg>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search players..."
            className="w-full pl-10 pr-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-colors"
          />
          {isSearching && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <LoadingSpinner size="sm" />
            </div>
          )}
          {searchQuery.length > 0 && !isSearching && (
            <button
              type="button"
              onClick={() => handleSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
              aria-label="Clear search"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
        {showingSearch && (
          <p className="text-xs text-slate-500 mt-1.5">
            {searchResults.length === 0
              ? 'No players found'
              : `${searchResults.length} player${searchResults.length !== 1 ? 's' : ''} found`}
          </p>
        )}
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900/50 text-slate-400 uppercase tracking-wider font-medium">
              <tr>
                <th className="px-6 py-4 w-16 text-center">{showingSearch ? '#' : 'Rank'}</th>
                <th className="px-6 py-4">Player</th>
                <th className="px-6 py-4 text-right">Rating</th>
                <th className="px-6 py-4 text-right">Win Rate</th>
                <th className="px-6 py-4 text-right">Games</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {displayUsers.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-8 text-center text-slate-500">
                    {showingSearch ? 'No players match your search' : 'No players yet'}
                  </td>
                </tr>
              )}
              {displayUsers.map((user, index) => {
                const gamesPlayed = user.gamesPlayed ?? 0;
                const gamesWon = user.gamesWon ?? 0;
                const winRate = gamesPlayed > 0 ? Math.round((gamesWon / gamesPlayed) * 100) : 0;
                const rank = index + 1;

                // Top 3 rank styling (only for leaderboard, not search)
                const rankBadge = showingSearch ? (
                  <span className="font-mono text-slate-500">{rank}</span>
                ) : rank === 1 ? (
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
                    <td className="px-6 py-4 font-medium">
                      <Link
                        to={`/player/${user.id}`}
                        className="text-white hover:text-emerald-400 transition-colors"
                      >
                        {user.username}
                      </Link>
                    </td>
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
