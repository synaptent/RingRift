import { Link, NavLink, Outlet } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Button } from './ui/Button';

const navLinkBaseClasses =
  'inline-flex items-center px-3 py-2 text-sm font-medium border-b-2 border-transparent';

function navLinkClasses({ isActive }: { isActive: boolean }) {
  return [
    navLinkBaseClasses,
    isActive
      ? 'border-emerald-500 text-emerald-300'
      : 'text-slate-300 hover:text-white hover:border-slate-500',
  ].join(' ');
}

export default function Layout() {
  const { user, logout } = useAuth();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-slate-900 focus:px-3 focus:py-2 focus:text-sm focus:text-white"
      >
        Skip to main content
      </a>

      <nav className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-6">
            <Link to="/" className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight text-white">RingRift</span>
            </Link>

            <div className="hidden gap-2 md:flex">
              <NavLink to="/" end className={navLinkClasses}>
                Home
              </NavLink>
              <NavLink to="/lobby" className={navLinkClasses}>
                Lobby
              </NavLink>
              <NavLink to="/leaderboard" className={navLinkClasses}>
                Leaderboard
              </NavLink>
              <NavLink to="/sandbox" className={navLinkClasses}>
                Sandbox
              </NavLink>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {user ? (
              <>
                <div className="hidden flex-col items-end text-xs sm:flex">
                  <span className="font-semibold text-slate-100">{user.username}</span>
                  {typeof user.rating === 'number' && (
                    <span className="text-slate-400">Rating {user.rating}</span>
                  )}
                </div>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={logout}
                  aria-label="Log out"
                >
                  Logout
                </Button>
              </>
            ) : (
              <Link
                to="/login"
                className="text-sm font-medium text-emerald-300 hover:text-emerald-200"
              >
                Login
              </Link>
            )}
          </div>
        </div>
      </nav>

      <main id="main-content" className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>
    </div>
  );
}
