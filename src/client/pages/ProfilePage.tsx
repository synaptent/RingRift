import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { authApi, gameApi, userApi, type GameSummary } from '../services/api';
import { User } from '../../shared/types/user';
import { Game, GameResult } from '../../shared/types/game';
import { ProfileSkeleton } from '../components/Skeleton';
import { InlineAlert } from '../components/ui/InlineAlert';
import { Button } from '../components/ui/Button';
import { extractErrorMessage } from '../utils/errorReporting';
import { formatVictoryReason } from '../adapters/gameViewModels';
import { RatingHistoryChart } from '../components/RatingHistoryChart';
import {
  computeAchievements,
  ACHIEVEMENT_DEFS,
  RARITY_COLORS,
  RARITY_TEXT,
  type UnlockedAchievement,
} from '../utils/achievements';

type RecentGame = Game & { resultReason?: GameResult['reason'] };

interface RatingHistoryEntry {
  date: string;
  rating: number;
  change: number;
  gameId: string | null;
}

interface BoardStats {
  boardType: string;
  label: string;
  played: number;
  won: number;
}

const BOARD_LABELS: Record<string, string> = {
  square8: 'Square 8x8',
  square19: 'Square 19x19',
  hex8: 'Hex Small',
  hexagonal: 'Hex Large',
};

export default function ProfilePage() {
  useDocumentTitle('Profile');
  const { user: currentUser } = useAuth();
  const [profile, setProfile] = useState<User | null>(null);
  const [recentGames, setRecentGames] = useState<RecentGame[]>([]);
  const [ratingHistory, setRatingHistory] = useState<RatingHistoryEntry[]>([]);
  const [allGames, setAllGames] = useState<GameSummary[]>([]);
  const [achievements, setAchievements] = useState<UnlockedAchievement[]>([]);
  const [boardStats, setBoardStats] = useState<BoardStats[]>([]);
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
      const userId = currentUser?.id;
      const [userData, gamesData, statsData, userGamesData] = await Promise.all([
        authApi.getProfile(),
        gameApi.getGames({ limit: 5 }).catch(() => ({ games: [] })),
        userApi.getStats().catch(() => ({ ratingHistory: [] })),
        userId
          ? gameApi.getUserGames(userId, { limit: 100, status: 'completed' }).catch(() => ({
              games: [],
              pagination: { total: 0, limit: 100, offset: 0, hasMore: false },
            }))
          : Promise.resolve({
              games: [] as GameSummary[],
              pagination: { total: 0, limit: 100, offset: 0, hasMore: false },
            }),
      ]);
      setProfile(userData);
      setRecentGames((gamesData.games ?? []) as RecentGame[]);
      setRatingHistory(statsData.ratingHistory ?? []);

      const completedGames = userGamesData.games ?? [];
      setAllGames(completedGames);

      // Compute per-board stats
      const statsMap = new Map<string, { played: number; won: number }>();
      for (const g of completedGames) {
        const bt = g.boardType;
        const entry = statsMap.get(bt) || { played: 0, won: 0 };
        entry.played++;
        if (g.winnerId === userData.id) entry.won++;
        statsMap.set(bt, entry);
      }
      const computed: BoardStats[] = [];
      for (const [bt, s] of statsMap) {
        computed.push({ boardType: bt, label: BOARD_LABELS[bt] ?? bt, ...s });
      }
      computed.sort((a, b) => b.played - a.played);
      setBoardStats(computed);

      // Compute achievements
      setAchievements(computeAchievements(userData, completedGames));

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
    return <ProfileSkeleton />;
  }

  if (!profile) {
    return (
      <div className="container mx-auto px-4 py-8 text-center text-red-400">
        {error || 'Profile not found'}
      </div>
    );
  }

  const totalAchievements = ACHIEVEMENT_DEFS.length;

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Profile header + stats */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
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

      {/* Per-board stats */}
      {boardStats.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
          <div className="p-6">
            <h2 className="text-lg font-bold text-white mb-4">Stats by Board</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {boardStats.map((bs) => {
                const wr = bs.played > 0 ? Math.round((bs.won / bs.played) * 100) : 0;
                return (
                  <div
                    key={bs.boardType}
                    className="p-3 bg-slate-900/50 rounded-lg border border-slate-700"
                  >
                    <div className="text-slate-300 text-sm font-medium mb-2">{bs.label}</div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-lg font-bold text-white">{bs.played}</span>
                      <span className="text-xs text-slate-500">games</span>
                    </div>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-sm font-semibold text-emerald-400">{wr}%</span>
                      <span className="text-xs text-slate-500">
                        win ({bs.won}W / {bs.played - bs.won}L)
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Achievements */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-white">Achievements</h2>
            <span className="text-sm text-slate-400">
              {achievements.length}/{totalAchievements}
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-900 rounded-full h-2 mb-5">
            <div
              className="bg-emerald-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${(achievements.length / totalAchievements) * 100}%` }}
            />
          </div>

          {achievements.length === 0 ? (
            <p className="text-slate-500 text-sm text-center py-4">
              Play games to unlock achievements!
            </p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {achievements.map((a) => (
                <div key={a.id} className={`p-3 rounded-lg border ${RARITY_COLORS[a.rarity]}`}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xl" role="img" aria-label={a.name}>
                      {a.icon}
                    </span>
                    <span className="text-sm font-semibold text-white truncate">{a.name}</span>
                  </div>
                  <p className="text-xs text-slate-400 leading-snug">{a.description}</p>
                  <span
                    className={`text-[10px] font-semibold uppercase tracking-wider mt-1.5 inline-block ${RARITY_TEXT[a.rarity]}`}
                  >
                    {a.rarity}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Locked achievements preview */}
          {achievements.length < totalAchievements && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <p className="text-xs text-slate-500 mb-2">Locked</p>
              <div className="flex flex-wrap gap-2">
                {ACHIEVEMENT_DEFS.filter((def) => !achievements.some((a) => a.id === def.id))
                  .slice(0, 6)
                  .map((def) => (
                    <div
                      key={def.id}
                      className="px-3 py-2 rounded-lg border border-slate-700/50 bg-slate-900/30 opacity-50"
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-base grayscale">{def.icon}</span>
                        <span className="text-xs text-slate-500">{def.name}</span>
                      </div>
                    </div>
                  ))}
                {totalAchievements - achievements.length > 6 && (
                  <div className="px-3 py-2 flex items-center">
                    <span className="text-xs text-slate-600">
                      +{totalAchievements - achievements.length - 6} more
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Games */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">Recent Games</h2>
        {allGames.length > 0 && (
          <Link
            to="/history"
            className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            View All Games
          </Link>
        )}
      </div>
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
                    <span className="text-slate-300 text-sm">
                      {BOARD_LABELS[game.boardType] ?? game.boardType}
                    </span>
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
