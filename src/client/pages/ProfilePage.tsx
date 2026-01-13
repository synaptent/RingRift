import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { authApi, gameApi, userApi } from '../services/api';
import { User } from '../../shared/types/user';
import { Game, GameResult } from '../../shared/types/game';
import LoadingSpinner from '../components/LoadingSpinner';
import { InlineAlert } from '../components/ui/InlineAlert';
import { Button } from '../components/ui/Button';
import { extractErrorMessage } from '../utils/errorReporting';
import { formatVictoryReason } from '../adapters/gameViewModels';
import { RatingHistoryChart } from '../components/RatingHistoryChart';

type RecentGame = Game & { resultReason?: GameResult['reason'] };

interface RatingHistoryEntry {
  date: string;
  rating: number;
  change: number;
  gameId: string | null;
}

export default function ProfilePage() {
  const { user: currentUser } = useAuth();
  const [profile, setProfile] = useState<User | null>(null);
  const [recentGames, setRecentGames] = useState<RecentGame[]>([]);
  const [ratingHistory, setRatingHistory] = useState<RatingHistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({ username: '' });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchProfileData();
  }, []);

  const fetchProfileData = async () => {
    try {
      setIsLoading(true);
      const [userData, gamesData, statsData] = await Promise.all([
        authApi.getProfile(),
        gameApi.getGames({ limit: 5 }).catch(() => ({ games: [] })),
        userApi.getStats().catch(() => ({ ratingHistory: [] })),
      ]);
      setProfile(userData);
      setRecentGames((gamesData.games ?? []) as RecentGame[]);
      setRatingHistory(statsData.ratingHistory ?? []);
      setEditForm({ username: userData.username ?? '' });
    } catch (err) {
      console.error('Failed to fetch profile:', err);
      setError('Failed to load profile data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const updatedUser = await authApi.updateProfile({ username: editForm.username });
      setProfile(updatedUser);
      setIsEditing(false);
      setError(null);
    } catch (error: unknown) {
      setError(extractErrorMessage(error, 'Failed to update profile'));
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="container mx-auto px-4 py-8 text-center text-red-400">
        {error || 'Profile not found'}
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-8">
        <div className="p-6 md:p-8">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                {isEditing ? 'Edit Profile' : profile.username}
              </h1>
              <p className="text-slate-400">
                Member since {new Date(profile.createdAt).toLocaleDateString()}
              </p>
            </div>
            {!isEditing && currentUser?.id === profile.id && (
              <Button type="button" variant="secondary" onClick={() => setIsEditing(true)}>
                Edit Profile
              </Button>
            )}
          </div>

          {isEditing ? (
            <form onSubmit={handleUpdateProfile} className="space-y-4 max-w-md">
              <div>
                <label
                  htmlFor="profile-username"
                  className="block text-sm font-medium text-slate-300 mb-1"
                >
                  Username
                </label>
                <input
                  id="profile-username"
                  type="text"
                  value={editForm.username}
                  onChange={(e) => setEditForm({ ...editForm, username: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-emerald-500"
                  minLength={3}
                  maxLength={20}
                />
              </div>
              {error && <InlineAlert variant="error">{error}</InlineAlert>}
              <div className="flex gap-3">
                <Button type="submit">Save Changes</Button>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => {
                    setIsEditing(false);
                    setEditForm({ username: profile.username });
                    setError(null);
                  }}
                >
                  Cancel
                </Button>
              </div>
            </form>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Rating</div>
                  <div className="text-2xl font-bold text-emerald-400">{profile.rating}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Games Played
                  </div>
                  <div className="text-2xl font-bold text-white">{profile.gamesPlayed}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Wins</div>
                  <div className="text-2xl font-bold text-white">{profile.gamesWon}</div>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Win Rate
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {profile.gamesPlayed > 0
                      ? `${Math.round((profile.gamesWon / profile.gamesPlayed) * 100)}%`
                      : '0%'}
                  </div>
                </div>
              </div>

              {/* Rating History Chart */}
              <div className="mt-6">
                <RatingHistoryChart history={ratingHistory} currentRating={profile.rating} />
              </div>
            </>
          )}
        </div>
      </div>

      <h2 className="text-xl font-bold text-white mb-4">Recent Games</h2>
      {recentGames.length === 0 ? (
        <div className="text-center py-8 text-slate-500 bg-slate-800/50 rounded-xl border border-slate-700">
          No games played yet
        </div>
      ) : (
        <div className="space-y-3">
          {recentGames.map((game) => {
            const isWinner = game.winnerId === profile.id;
            const isDraw = game.status === 'completed' && !game.winnerId;
            const reasonLabel = game.resultReason ? formatVictoryReason(game.resultReason) : null;

            return (
              <div
                key={game.id}
                className="p-4 bg-slate-800 rounded-lg border border-slate-700 flex items-center justify-between hover:border-slate-600 transition-colors"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className={`text-sm font-bold ${
                        isWinner ? 'text-emerald-400' : isDraw ? 'text-slate-400' : 'text-red-400'
                      }`}
                    >
                      {isWinner ? 'Victory' : isDraw ? 'Draw' : 'Defeat'}
                    </span>
                    <span className="text-slate-500 text-xs">•</span>
                    <span className="text-slate-300 text-sm">{game.boardType}</span>
                    {reasonLabel && (
                      <>
                        <span className="text-slate-500 text-xs">•</span>
                        <span className="text-slate-400 text-xs">{reasonLabel}</span>
                      </>
                    )}
                  </div>
                  <div className="text-xs text-slate-500">
                    {new Date(game.createdAt).toLocaleDateString()}
                  </div>
                </div>
                <Link
                  to={`/game/${game.id}`}
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs font-medium rounded transition-colors"
                >
                  View Game
                </Link>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
