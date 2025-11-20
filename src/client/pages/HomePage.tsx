import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-10 space-y-8">
      <header className="space-y-2">
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-50">
          Welcome to RingRift
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl">
          You're signed in. From here you can join the lobby to create backend games,
          explore the rules in the local sandbox, or inspect your profile and the leaderboard.
        </p>
      </header>

      {/* Primary actions */}
      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Link
          to="/lobby"
          className="group relative overflow-hidden rounded-2xl border border-emerald-600/70 bg-gradient-to-br from-emerald-700 to-emerald-500 px-5 py-6 shadow-lg hover:shadow-emerald-500/40 transition-shadow"
        >
          <div className="flex flex-col gap-2">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              Enter Lobby
              <span className="inline-flex items-center justify-center rounded-full bg-emerald-900/70 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-200">
                Backend
              </span>
            </h2>
            <p className="text-sm text-emerald-50/90">
              Create or join backend games, match with other players or AIs, and jump into a live
              game at <code className="font-mono text-xs">/game/:gameId</code>.
            </p>
          </div>
        </Link>

        <Link
          to="/sandbox"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-emerald-500/70 hover:bg-slate-900 transition-colors"
        >
          <h2 className="text-lg font-semibold text-slate-100">Open Local Sandbox</h2>
          <p className="mt-1 text-sm text-slate-300">
            Run the full rules engine in your browser only. Ideal for testing movement, captures,
            lines, and territory without touching the backend.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            From the sandbox you can also attempt a backend-backed sandbox game via
            &nbsp;<span className="font-mono">Launch Game</span>.
          </p>
        </Link>

        <Link
          to="/leaderboard"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-amber-500/70 hover:bg-slate-900 transition-colors"
        >
          <h2 className="text-lg font-semibold text-slate-100">View Leaderboard</h2>
          <p className="mt-1 text-sm text-slate-300">
            Inspect rated results and player ratings backed by the database and rating service.
          </p>
        </Link>

        <Link
          to="/profile"
          className="group rounded-2xl border border-slate-700 bg-slate-900/70 px-5 py-6 hover:border-sky-500/70 hover:bg-slate-900 transition-colors"
        >
          <h2 className="text-lg font-semibold text-slate-100">Profile & Settings</h2>
          <p className="mt-1 text-sm text-slate-300">
            View your account details and (as implemented) tweak preferences and inspect your
            game history.
          </p>
        </Link>

        <a
          href="/sandbox"
          className="group rounded-2xl border border-slate-800 bg-slate-950/60 px-5 py-6 hover:border-fuchsia-500/60 hover:bg-slate-950 transition-colors"
        >
          <h2 className="text-lg font-semibold text-slate-100">Rules & Diagnostics</h2>
          <p className="mt-1 text-sm text-slate-300">
            Combine <code className="font-mono text-xs">/sandbox</code> with the parity and
            scenario tests in the repo to validate rules behaviour against the backend GameEngine.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            See <code className="font-mono">tests/scenarios</code> and
            &nbsp;<code className="font-mono">tests/unit</code> for the full matrix.
          </p>
        </a>
      </section>

      {/* Backend quick links */}
      <section className="space-y-2">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
          Backend & API shortcuts
        </h2>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3 text-sm text-slate-300">
          <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3">
            <p className="font-medium text-slate-100 mb-1">Health & routes</p>
            <p className="text-xs text-slate-400 mb-1">
              Backend HTTP API is served under <code className="font-mono">/api</code>.
            </p>
            <p className="text-xs text-slate-500">
              Try <code className="font-mono">GET /api</code> or run the health tests in
              &nbsp;<code className="font-mono">tests/unit/server.health-and-routes.test.ts</code>.
            </p>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3">
            <p className="font-medium text-slate-100 mb-1">Games API</p>
            <p className="text-xs text-slate-400 mb-1">
              Lobby and game creation use <code className="font-mono">/api/games</code>.
            </p>
            <p className="text-xs text-slate-500">
              Creating or joining games here will route you to
              &nbsp;<code className="font-mono">/game/:gameId</code> with WebSocket updates.
            </p>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3">
            <p className="font-medium text-slate-100 mb-1">WebSockets</p>
            <p className="text-xs text-slate-400 mb-1">
              Live games use the WebSocket server on port 3001.
            </p>
            <p className="text-xs text-slate-500">
              See <code className="font-mono">src/server/websocket/server.ts</code> and the
              integration tests in
              &nbsp;<code className="font-mono">tests/unit/WebSocketServer.aiTurn.integration.test.ts</code>.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
