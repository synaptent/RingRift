import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-10 space-y-8">
      <header className="space-y-2">
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-50 flex items-center gap-3">
          <img src="/ringrift-icon.png" alt="RingRift" className="w-10 h-10" />
          Welcome to RingRift
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl">
          You’re signed in. From here you can join the lobby to create backend games, explore the
          rules in the local sandbox, or inspect your profile and the leaderboard.
        </p>
      </header>

      {/* Primary actions */}
      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Link
          to="/lobby"
          className="group relative overflow-hidden rounded-2xl border border-emerald-600/70 bg-gradient-to-br from-emerald-700 to-emerald-500 px-5 py-6 shadow-lg hover:shadow-emerald-500/40 hover:scale-[1.02] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <div className="flex flex-col gap-2">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
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
          <h2 className="text-lg font-semibold text-slate-100">Open Local Sandbox</h2>
          <p className="mt-1 text-sm text-slate-300">
            Play offline in your browser. Perfect for practicing movement, captures, lines, and
            territory scoring.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            You can also start an online game from the sandbox using the Launch Game button.
          </p>
        </Link>

        <Link
          to="/sandbox?preset=learn-basics"
          className="group rounded-2xl border border-emerald-500/40 bg-slate-900/70 px-5 py-6 hover:border-emerald-400/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <div className="flex items-start justify-between gap-3">
            <h2 className="text-lg font-semibold text-slate-100">Learn the Basics</h2>
            <span className="inline-flex items-center rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-200 border border-emerald-500/30">
              Tutorial
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-300">
            Jump straight into a guided starter match. Great for first-time players who want to
            learn placement, movement, and captures quickly.
          </p>
        </Link>

        <Link
          to="/leaderboard"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-amber-500/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-amber-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <h2 className="text-lg font-semibold text-slate-100">View Leaderboard</h2>
          <p className="mt-1 text-sm text-slate-300">
            Inspect rated results and player ratings backed by the database and rating service.
          </p>
        </Link>

        <Link
          to="/profile"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-sky-500/70 hover:bg-slate-900 hover:shadow-lg hover:scale-[1.01] transition-all duration-200 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
        >
          <h2 className="text-lg font-semibold text-slate-100">Profile & Settings</h2>
          <p className="mt-1 text-sm text-slate-300">
            View your account details, game history, rating progress, and preferences.
          </p>
        </Link>
      </section>
    </div>
  );
}
