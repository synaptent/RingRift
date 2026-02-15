import { useState, useEffect } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Button } from './ui/Button';
import { ButtonLink } from './ui/ButtonLink';
import { SettingsModal } from './SettingsModal';

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

const mobileNavLinkClasses = ({ isActive }: { isActive: boolean }) =>
  [
    'block px-4 py-3 text-base font-medium rounded-lg transition-colors',
    isActive
      ? 'bg-emerald-500/20 text-emerald-300'
      : 'text-slate-300 hover:bg-slate-800 hover:text-white',
  ].join(' ');

export default function Layout() {
  const { user, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const location = useLocation();

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  // Global keyboard shortcut for settings (Ctrl/Cmd + ,)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === ',') {
        e.preventDefault();
        setSettingsOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 app-bg">
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
              <img src="/ringrift-icon.png" alt="" className="h-8 w-8" aria-hidden="true" />
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
              <NavLink to="/history" className={navLinkClasses}>
                History
              </NavLink>
              <NavLink to="/sandbox" className={navLinkClasses}>
                Practice
              </NavLink>
              <NavLink to="/profile" className={navLinkClasses}>
                Profile
              </NavLink>
              <NavLink to="/help" className={navLinkClasses}>
                Help
              </NavLink>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Settings button */}
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="p-2 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 transition-colors"
              aria-label="Open settings"
              title="Settings (Ctrl+,)"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            </button>

            {user ? (
              <>
                <Link
                  to="/profile"
                  className="hidden flex-col items-end text-xs sm:flex hover:opacity-80 transition-opacity"
                >
                  <span className="font-semibold text-slate-100">{user.username}</span>
                  {typeof user.rating === 'number' && (
                    <span className="text-slate-400">Rating {user.rating}</span>
                  )}
                </Link>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={logout}
                  className="hidden sm:inline-flex"
                >
                  Logout
                </Button>
              </>
            ) : (
              <Link
                to="/login"
                className="hidden text-sm font-medium text-emerald-300 hover:text-emerald-200 sm:inline"
              >
                Login
              </Link>
            )}

            {/* Mobile hamburger button */}
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-md p-2 text-slate-400 hover:bg-slate-800 hover:text-white focus:outline-none focus:ring-2 focus:ring-inset focus:ring-emerald-500 md:hidden"
              aria-expanded={mobileMenuOpen}
              aria-controls="mobile-menu"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <span className="sr-only">{mobileMenuOpen ? 'Close menu' : 'Open menu'}</span>
              {mobileMenuOpen ? (
                <svg
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu dropdown */}
        {mobileMenuOpen && (
          <div className="border-t border-slate-800 md:hidden" id="mobile-menu">
            <div className="space-y-1 px-3 py-4">
              <NavLink to="/" end className={mobileNavLinkClasses}>
                Home
              </NavLink>
              <NavLink to="/lobby" className={mobileNavLinkClasses}>
                Lobby
              </NavLink>
              <NavLink to="/leaderboard" className={mobileNavLinkClasses}>
                Leaderboard
              </NavLink>
              <NavLink to="/history" className={mobileNavLinkClasses}>
                History
              </NavLink>
              <NavLink to="/sandbox" className={mobileNavLinkClasses}>
                Practice
              </NavLink>
              <NavLink to="/profile" className={mobileNavLinkClasses}>
                Profile
              </NavLink>
              <NavLink to="/help" className={mobileNavLinkClasses}>
                Help
              </NavLink>
            </div>

            {/* Mobile user section */}
            <div className="border-t border-slate-800 px-3 py-4">
              {user ? (
                <div className="flex items-center justify-between">
                  <Link to="/profile" className="flex flex-col">
                    <span className="text-sm font-semibold text-slate-100">{user.username}</span>
                    {typeof user.rating === 'number' && (
                      <span className="text-xs text-slate-400">Rating {user.rating}</span>
                    )}
                  </Link>
                  <Button type="button" variant="secondary" size="sm" onClick={logout}>
                    Logout
                  </Button>
                </div>
              ) : (
                <ButtonLink to="/login" className="w-full justify-center">
                  Login
                </ButtonLink>
              )}
            </div>
          </div>
        )}
      </nav>

      <main id="main-content" className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>

      {/* Settings modal */}
      <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
