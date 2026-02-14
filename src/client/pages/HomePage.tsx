import { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useDocumentTitle } from '../hooks/useDocumentTitle';

function ChallengeLink() {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    const url = `${window.location.origin}/sandbox?preset=sq8-1h-1ai`;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = url;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="group rounded-2xl border border-sky-500/40 bg-slate-900/70 px-5 py-6 hover:border-sky-400/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 text-left w-full"
    >
      <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
        <svg
          className="w-5 h-5 text-sky-400/70"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M7.217 10.907a2.25 2.25 0 100 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186l9.566-5.314m-9.566 7.5l9.566 5.314m0 0a2.25 2.25 0 103.935 2.186 2.25 2.25 0 00-3.935-2.186zm0-12.814a2.25 2.25 0 103.933-2.185 2.25 2.25 0 00-3.933 2.185z"
          />
        </svg>
        Challenge a Friend
      </h2>
      <p className="mt-1 text-sm text-slate-300">
        {copied ? (
          <span className="text-sky-400 font-medium">Link copied! Send it to a friend.</span>
        ) : (
          'Copy a game link and send it to anyone. No account needed to play.'
        )}
      </p>
    </button>
  );
}

export default function HomePage() {
  const { user } = useAuth();
  useDocumentTitle('Home');

  return (
    <div className="container mx-auto px-4 py-10 space-y-8">
      <header className="space-y-2">
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-50 flex items-center gap-3">
          <img src="/ringrift-icon.png" alt="RingRift" className="w-10 h-10" />
          Welcome{user?.username ? `, ${user.username}` : ' to RingRift'}
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl">
          Play online matches, practice against AI, or check out the leaderboard.
        </p>
      </header>

      {/* Quick play banner */}
      <Link
        to="/sandbox?preset=learn-basics"
        className="block rounded-2xl border border-emerald-500/50 bg-gradient-to-r from-emerald-900/40 to-sky-900/30 px-6 py-5 hover:border-emerald-400/70 hover:shadow-lg hover:shadow-emerald-500/10 hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
      >
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              Play Now
              <svg
                className="w-5 h-5 text-emerald-400"
                fill="currentColor"
                viewBox="0 0 20 20"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                  clipRule="evenodd"
                />
              </svg>
            </h2>
            <p className="mt-1 text-sm text-slate-300">
              Jump into a game instantly. No setup needed.
            </p>
          </div>
          <svg
            className="w-6 h-6 text-slate-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </Link>

      {/* Primary actions */}
      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Link
          to="/lobby"
          className="group relative overflow-hidden rounded-2xl border border-emerald-600/70 bg-gradient-to-br from-emerald-700 to-emerald-500 px-5 py-6 shadow-lg hover:shadow-emerald-500/20 hover:scale-[1.02] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <div className="flex flex-col gap-2">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <svg
                className="w-5 h-5 opacity-80"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418"
                />
              </svg>
              Enter Lobby
              <span className="inline-flex items-center justify-center rounded-full bg-emerald-900/70 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-200">
                Online
              </span>
            </h2>
            <p className="text-sm text-emerald-50/90">
              Create or join online games, match with other players or AI opponents, and jump into
              live multiplayer matches.
            </p>
          </div>
        </Link>

        <Link
          to="/sandbox"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-emerald-500/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
            <svg
              className="w-5 h-5 text-slate-400 group-hover:text-emerald-400 transition-colors"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
              />
            </svg>
            Practice Mode
          </h2>
          <p className="mt-1 text-sm text-slate-300">
            Play offline. Try strategies against AI or practice with a friend.
          </p>
        </Link>

        <ChallengeLink />

        <Link
          to="/leaderboard"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-amber-500/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
            <svg
              className="w-5 h-5 text-slate-400 group-hover:text-amber-400 transition-colors"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M18.75 4.236c.982.143 1.954.317 2.916.52A6.003 6.003 0 0016.27 9.728M18.75 4.236V4.5c0 2.108-.966 3.99-2.48 5.228m0 0a6.003 6.003 0 01-3.77 1.522m0 0a6.003 6.003 0 01-3.77-1.522"
              />
            </svg>
            View Leaderboard
          </h2>
          <p className="mt-1 text-sm text-slate-300">
            See top players and track your rating progress.
          </p>
        </Link>

        <Link
          to="/profile"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-sky-500/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
            <svg
              className="w-5 h-5 text-slate-400 group-hover:text-sky-400 transition-colors"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
              />
            </svg>
            Profile & Settings
          </h2>
          <p className="mt-1 text-sm text-slate-300">Your game history, stats, and settings.</p>
        </Link>
      </section>
    </div>
  );
}
